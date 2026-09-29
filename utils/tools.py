
import os
from collections import defaultdict, deque
import datetime
import torch
import torch.distributed as dist

def close_ddp_env():
    dist.destroy_process_group()
    torch.cuda.empty_cache()

def init_ddp_env(backend='nccl'):
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(days=1))
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return world_size, global_rank, local_rank

def format_time_hms(current_time, start_time=0):
    total_seconds = current_time - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return total_seconds, f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

def save_checkpoint(iter, model, optimizer, lr_scheduler, 
                    save_root, ckpt_name, save_model_only=True):
    
    os.makedirs(save_root, exist_ok=True)
    if save_model_only:
        model_dict = model.state_dict()
        torch.save(model_dict, os.path.join(save_root, f"Model_{ckpt_name}.pth"))
    else:
        ckpt_dict = {
            'iter': iter,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }
        torch.save(ckpt_dict, os.path.join(save_root, f"{ckpt_name}.pth"))

class SmoothedValue(object):
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return (f"median: {self.median}, "
                f"avg: {self.avg}, "
                f"global_avg: {self.global_avg}, "
                f"max: {self.max}, "
                f"value: {self.value}, ")
    
