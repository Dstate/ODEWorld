import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.optim.lr_scheduler import SequentialLR, LinearLR

def CosineAnnealingWarmUpRestarts(optimizer, T_max, T_warmup=2000, start_factor=0.1, eta_min=1e-4):
       warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=T_warmup)
       annealing_scheduler = CosineAnnealingLR(optimizer, T_max=T_max - T_warmup, eta_min=eta_min)
       scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, annealing_scheduler], milestones=[T_warmup])
       return scheduler