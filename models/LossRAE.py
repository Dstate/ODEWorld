from torch import nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.optim.lr_scheduler import SequentialLR, LinearLR
import math
from .disc import LPIPS
import torch.nn.functional as F
from torch.optim import AdamW
from .disc.diffaug import DiffAug
from .disc.discriminator import DinoDiscriminator
from .disc.gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss

def CosineAnnealingWarmUpRestarts(optimizer, T_max, T_warmup=2000, start_factor=0.1, eta_min=1e-4):
       warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=T_warmup)
       annealing_scheduler = CosineAnnealingLR(optimizer, T_max=T_max - T_warmup, eta_min=eta_min)
       scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, annealing_scheduler], milestones=[T_warmup])
       return scheduler


def prepare_discriminator(
    disc_ckpt_path = 'assets/weights/RAE/dino_vit_small_patch8_224.pth',
    ks = 9,
    key_depths = (2, 5, 8, 11),
    norm_type = 'bn',
    using_spec_norm = True,
    norm_eps = 1e-6,
    recipe = 'S_8',
    aug_prob = 1.0,
    aug_cutout = 0.0
) -> tuple[DinoDiscriminator, DiffAug]:
    
    disc = DinoDiscriminator(
        device=torch.device('cuda'),
        dino_ckpt_path=disc_ckpt_path,
        ks=ks,
        key_depths=key_depths,
        norm_type=norm_type,
        using_spec_norm=using_spec_norm,
        norm_eps=norm_eps,
        recipe=recipe,
    ).cuda()

    augment = DiffAug(prob=aug_prob, cutout=aug_cutout)

    return disc, augment


def prepare_disc_training(
    disc_params, 
    gan_total_iters,
    gan_warmup_iters = 2000,
    gan_eta_min_lr = 1e-5,
    learning_rate=3e-4,  
    weight_decay = 0.0, 
):
    optimizer = AdamW(disc_params, lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_warmup=gan_warmup_iters, 
                                                 T_max=gan_total_iters, eta_min=gan_eta_min_lr)

    return optimizer, lr_scheduler


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()



class LossFunc(object):
    def __init__(
        self,
        lpips_weight = 1.0,
        gan_weight = 0.75,
        gan_disc_ckpt_path = 'assets/weights/RAE/dino_vit_small_patch8_224.pth',
        num_iters = 200000,
        gan_loss_percent = 0.50,
        gan_train_percent = 0.75,
        
    ):
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        # Let rank 0 populate the shared VGG16 and LPIPS caches first.
        if distributed and torch.distributed.get_rank() != 0:
            torch.distributed.barrier()
        self.lpips = LPIPS().cuda()
        if distributed and torch.distributed.get_rank() == 0:
            torch.distributed.barrier()
        self.lpips.eval()
        self.lpips_weight = lpips_weight
        self.gan_weight = gan_weight

        # gan setting
        self.num_iters = num_iters
        self.gan_loss_iters = int(num_iters * gan_loss_percent)
        self.gan_train_iters = int(num_iters * gan_train_percent)
        # build disc
        self.discriminator, self.disc_aug = prepare_discriminator(disc_ckpt_path=gan_disc_ckpt_path)
        disc_params = [p for p in self.discriminator.parameters() if p.requires_grad]
        # disc training
        self.disc_optimizer, self.disc_scheduler = prepare_disc_training(disc_params=disc_params, gan_total_iters=self.gan_train_iters)
        self.disc_loss_fn = hinge_d_loss
        self.gen_loss_fn = vanilla_g_loss

    def recon_base_loss(self, obs, obs_rec):
        loss_base = F.l1_loss(obs, obs_rec)
        return loss_base

    def recon_lpips_loss(self, obs, obs_rec):
        loss_lpips = self.lpips(obs, obs_rec)
        return loss_lpips
    
    def gan_loss(self, obs_rec):
        self.discriminator.eval()
        obs_rec_normed = obs_rec * 2.0 - 1.0
        fake_aug = self.disc_aug.aug(obs_rec_normed)
        # with torch.no_grad():
        logits_fake, _ = self.discriminator(fake_aug, None)
        gan_loss = self.gen_loss_fn(logits_fake)
        return gan_loss

    def gan_train(self, real_obs, fake_obs):
        if self.num_iters <= self.gan_train_iters:
            self.discriminator.train()
            real_obs_norm = real_obs * 2.0 - 1.0
            fake_obs_norm = (fake_obs * 2.0 - 1.0).clamp(-1.0, 1.0)
            # forward
            self.disc_optimizer.zero_grad(set_to_none=True)
            fake_obs_norm = torch.round((fake_obs_norm + 1.0) * 127.5) / 127.5 - 1.0
            fake_input = self.disc_aug.aug(fake_obs_norm)
            real_input = self.disc_aug.aug(real_obs_norm)
            logits_fake, logits_real = self.discriminator(fake_input, real_input)
            d_loss = self.disc_loss_fn(logits_real, logits_fake)
            # backward
            d_loss.backward()
            self.disc_optimizer.step()
            self.disc_scheduler.step()

            gan_train_dict = dict(disc_loss=d_loss.detach(),
                logits_real=logits_real.detach().mean(),
                logits_fake=logits_fake.detach().mean(),
                disc_lr=self.disc_scheduler.get_last_lr()[0])
        else :
            gan_train_dict = dict(disc_loss=0,
                logits_real=0,
                logits_fake=0,
                disc_lr=0)

        return gan_train_dict

    def __call__(self, obs, obs_rec, last_layer=None):
        # recon loss
        loss_base = self.recon_base_loss(obs, obs_rec)
        loss_lpips = self.recon_lpips_loss(obs, obs_rec)
        loss_recon = loss_base + loss_lpips * self.lpips_weight
        # add gan loss
        if self.num_iters <= self.gan_loss_iters:
            loss_gan = self.gan_loss(obs_rec)
            ada_gan_weight = 1.0 if last_layer is None else calculate_adaptive_weight(loss_recon, loss_gan, last_layer)
        else:
            loss_gan = torch.zeros_like(loss_base)
            ada_gan_weight = 0.0

        loss_total = loss_recon + ada_gan_weight * self.gan_weight * loss_gan

        self.num_iters -= 1

        loss_dict = dict(
            loss_base=loss_base,
            loss_lpips=loss_lpips,
            loss_gan=loss_gan,
            ada_gan_weight=ada_gan_weight)

        return loss_total, loss_dict
    
    
def build_loss_func(
    lpips_weight = 1.0,
    gan_weight = 0.75,
    gan_disc_ckpt_path = 'assets/weights/RAE/dino_vit_small_patch8_224.pth',
    num_iters = 200000,
    **kwargs):

    loss_func = LossFunc(lpips_weight=lpips_weight,
                         gan_weight=gan_weight,
                         gan_disc_ckpt_path=gan_disc_ckpt_path,
                         num_iters=num_iters)
    
    return loss_func
