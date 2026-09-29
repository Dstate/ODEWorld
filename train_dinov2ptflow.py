import os
import math
import time
import torch
import random
import argparse
import json
import numpy as np
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils import RoboModelWrapper
from utils import Logger, SmoothedValue
from utils import EPDataLoaderWithTimeWrapper
from utils import py2dict, merge_dict_to_dict, format_time_hms
from utils import init_ddp_env, close_ddp_env
from utils import CosineAnnealingWarmUpRestarts
from datetime import datetime

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_base_config():
    config = dict(
        # base settings
        logger_type = "tensorboard",
        # model_config 
        encoder_config_path = 'assets/weights/dinov2-with-registers-base',
        disable_encoder_detach = False,
        # engine_config
        image_size=256,
        batch_size=2,
        num_workers=8,
        max_time_length=50, 
        vel_chunk_length_half=2,
        data_downsample_ratio=1,
        sample_per_traj=4,
        # training_config
        num_iters = 100000,
        rec_warm_iters = 0,
        save_interval = 20000,
        log_interval = 20,
        learning_rate = 1e-4,
        weight_decay = 0,
        warm_steps = 2000,
        eta_min_lr = 0,
        v_decay_start_frac = 0.8,
    )

    return config


def get_args_parser():
    parser = argparse.ArgumentParser('training script', add_help=False)
    parser.add_argument('--meta_file_path', type=str, default='assets/metas/libero_train.json')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_iters', type=int, default=100000)
    parser.add_argument('--max_time_length', type=int, default=50)
    parser.add_argument('--data_downsample_ratio', type=int, default=1)
    parser.add_argument('--disable_encoder_detach', action="store_true", default=False)

    cfg = parser.parse_args()
    cfg.output_dir = os.path.join(cfg.output_dir, datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    config = get_base_config()
    config = merge_dict_to_dict(vars(cfg), config)
    return config


def train(config, logger, model, train_loader, flag='ckpt'):
    start_time = time.time()
    loss_recod = SmoothedValue()
    
    model = RoboModelWrapper(model)
    model = DistributedDataParallel(model)
    train_loader = EPDataLoaderWithTimeWrapper(train_loader, total_iters=config['num_iters'])

    v_keys = ('v_model', 't_process')
    z_params, v_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (v_params if any(k in name for k in v_keys) else z_params).append(p)
    optimizer = AdamW([
        {'params': z_params, 'lr': config['learning_rate']},
        {'params': v_params, 'lr': config['learning_rate']},
    ], weight_decay=config['weight_decay'])

    warm, total = config['warm_steps'], config['num_iters']
    v_decay_start = int(total * config['v_decay_start_frac'])

    def z_lambda(step):
        if step < warm:
            return step / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * t))

    def v_lambda(step):
        if step < warm:
            return step / max(1, warm)
        if step < v_decay_start:
            return 1.0
        t = (step - v_decay_start) / max(1, total - v_decay_start)
        return 0.5 * (1 + math.cos(math.pi * t))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[z_lambda, v_lambda])
    
    for ep_iter, (batch, data_time) in enumerate(train_loader):
        iter_start_time = time.time()
        
        # calculate loss and update model
        optimizer.zero_grad(set_to_none=True)
        loss, loss_metric = model(**batch, rec_warm=(ep_iter < config['rec_warm_iters']))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        
        # log training message
        loss_recod.update(loss.detach().cpu().item())
        loss_metric = {k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v)
                       for k, v in loss_metric.items()}
        loss_metric.update({'iter': ep_iter,
                            'smoothed_loss': loss_recod.avg,
                            'lr': lr_scheduler.get_last_lr()[0],
                            'lr_v': lr_scheduler.get_last_lr()[1]})
        logger.log_metric(loss_metric)
        if ep_iter % config['log_interval'] == 0 or ep_iter == len(train_loader)-1:
            current_time = time.time()
            elapsed_time_sec, elapsed_time = format_time_hms(current_time, start_time)
            iter_time_sec, iter_time = format_time_hms(current_time, iter_start_time)
            data_time_sec, data_time = format_time_hms(data_time)
            eta_sec, eta = format_time_hms(iter_time_sec * (len(train_loader)-ep_iter-1))
            logger.log_msg_every(
                f"{flag} Iter [{ep_iter}/{len(train_loader)}], "
                + ", ".join([f"{k}: {v:.6f}" for k, v in loss_metric.items()])
                + f", iter_time: {iter_time_sec:.4f}s, "
                + f"data_time: {data_time_sec:.4f}s, "
                + f"elapsed_time: {elapsed_time}, "
                + f"eta: {eta}")
        
        
        # save checkpoint
        if (ep_iter + 1) % config['save_interval'] == 0 or (ep_iter + 1) == len(train_loader):
            logger.log_msg(f"Saving ckpt : [{ep_iter + 1}]")
            unwraped_model = model.module.unwrap()
            logger.save_checkpoint(ep_iter, unwraped_model, optimizer, lr_scheduler, 
                                  f"{flag}_{ep_iter + 1}", True)
            logger.log_msg(f"Save ok")
    
    return unwraped_model


def prepare_training_components(config):

    from dataloader.ImgVelEngine import build_vel_dataloader
    from models.DINOv2PTFlow import Dinov2PTflowImgoal

    train_loader = build_vel_dataloader(**config)
    model = Dinov2PTflowImgoal(**config)
    return model, train_loader


def main_ddp(config):
    world_size, global_rank, local_rank = init_ddp_env()
    config['world_size'] = world_size
    config['global_rank'] = global_rank
    config['local_rank'] = local_rank
    
    seed_everything(config['seed'])
    logger = Logger(**config)
    logger.log_msg_every("init env ok")
    logger.log_msg_every(f"current device is: {torch.cuda.current_device()}")    
    logger.log_msg_every(f"current seed is: {config['seed']}")
    
    # training
    model, train_loader = prepare_training_components(config)
    model = train(config, logger, model, train_loader, flag='ckpt')
    close_ddp_env()
    
if __name__ == '__main__':
    config = get_args_parser()
    main_ddp(config)
    