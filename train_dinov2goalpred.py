import os
import time
import torch
import random
import argparse
import numpy as np
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
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
        num_layers = 4,
        num_lang_tokens = 1,
        # engine_config
        image_size=256,
        batch_size=16,
        num_workers=8,
        max_time_length=50,
        vel_chunk_length_half=2,
        data_downsample_ratio=1,
        sample_per_traj=1,
        # training_config
        num_iters = 100000,
        save_interval = 20000,
        log_interval = 20,
        learning_rate = 3e-4,
        weight_decay = 0,
        warm_steps = 2000,
        eta_min_lr = 0
    )

    return config

def get_args_parser():
    parser = argparse.ArgumentParser('training script', add_help=False)
    parser.add_argument('--meta_file_path', type=str, default='assets/metas/libero.json')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--seed', type=int, default=42)
    cfg = parser.parse_args()
    cfg.output_dir = os.path.join(cfg.output_dir, datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    config = get_base_config()
    config = merge_dict_to_dict(vars(cfg), config)
    return config



def train(config, logger, model, train_loader, flag='ckpt'):
    start_time = time.time()
    loss_recod = SmoothedValue()
    
    model = model.to('cuda', torch.float32)
    model = DistributedDataParallel(model)
    train_loader = EPDataLoaderWithTimeWrapper(train_loader, total_iters=config['num_iters'])
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_warmup=config['warm_steps'], 
                                    T_max=config['num_iters'], eta_min=config['eta_min_lr'])
    
    for ep_iter, (batch, data_time) in enumerate(train_loader):
        iter_start_time = time.time()
        obs_s0 = batch['obs_s0'].to('cuda', torch.float32)
        obs_sg = batch['obs_sg'].to('cuda', torch.float32)
        lang = batch['lang']

        # calculate loss and update model
        optimizer.zero_grad(set_to_none=True)
        loss, loss_metric = model(obs_s0, obs_sg, lang)
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
                            'lr': lr_scheduler.get_last_lr()[0]})
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
            unwraped_model = model.module
            logger.save_checkpoint(ep_iter, unwraped_model, optimizer, lr_scheduler, 
                                  f"{flag}_{ep_iter + 1}", True)
            logger.log_msg(f"Save ok")
    
    return unwraped_model


def prepare_training_components(config):

    from dataloader.ImgVelEngine import build_vel_dataloader
    from models.DINOv2GoalPred import build_dino_goal_pred

    train_loader = build_vel_dataloader(**config)
    model = build_dino_goal_pred(**config)
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
