import os
import os.path as osp
import torch
import json 
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import h5py
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class RandomCropRatio(A.ImageOnlyTransform):
    def __init__(self, ratio: float = 0.95, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        if not 0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = ratio

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("ratio",)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        new_h, new_w = int(h*self.ratio), int(w*self.ratio)
        y_min = np.random.randint(0, h - new_h + 1)
        x_min = np.random.randint(0, w - new_w + 1)
        y_max = y_min + new_h
        x_max = x_min + new_w
        return img[y_min:y_max, x_min:x_max]


def build_base_transform(n_px, aug=True, 
                        crop_scale=0.95, crop_prob=1.0):

    base_transform = []
    base_transform.append(A.Resize(height=n_px, width=n_px))
    
    if aug:
        base_transform.append(RandomCropRatio(ratio=crop_scale, p=crop_prob))
    else :
        base_transform.append(A.CenterCrop(height=int(n_px*crop_scale), width=int(n_px*crop_scale), p=crop_prob))    
        
    base_transform.append(A.Resize(height=n_px, width=n_px))
    base_transform.append(A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)))
    base_transform.append(ToTensorV2())
    base_transform = A.ReplayCompose(base_transform)
    return base_transform


class Processor(object):
    def __init__(self, meta_file_path, image_size=256, eps=1e-6, training=True):
        assert osp.isfile(meta_file_path), 'dataset statistics don\'t exit'
        dataset_statistics = json.load(open(meta_file_path, 'r'))
        self.dataset_statistics = dataset_statistics
        self.action_max = np.array(dataset_statistics['action_max'])
        self.action_min = np.array(dataset_statistics['action_min'])
        self.proprio_max = np.array(dataset_statistics['proprio_max'])
        self.proprio_min = np.array(dataset_statistics['proprio_min'])
        self.img_transform = build_base_transform(image_size, aug=training)
        
        self.eps = eps
        self.action_length = len(dataset_statistics['action_max'])
        self.proprio_length = len(dataset_statistics['proprio_max'])

    def preprocess_action(self, action):
        action = np.clip(action, a_max=self.action_max, a_min=self.action_min)
        action = (action - self.action_min) / (self.action_max - self.action_min + self.eps) * 2 - 1
        action = torch.from_numpy(action)
        return action
    
    def preprocess_proprio(self, proprio):
        proprio = np.clip(proprio, a_max=self.proprio_max, a_min=self.proprio_min)
        proprio = (proprio - self.proprio_min) / (self.proprio_max - self.proprio_min + self.eps) * 2 - 1
        proprio = torch.from_numpy(proprio) 
        return proprio
    
    def preprocess_image(self, img, replay_params=None):
        if replay_params == None:
            transformed = self.img_transform(image=img)
            transformed_image = transformed['image']
            replay_params = transformed['replay']
        else :
            transformed = A.ReplayCompose.replay(replay_params, image=img)
            transformed_image = transformed['image']
        return transformed_image, replay_params

    def postprocess_action(self, action):
        action = action[..., :self.action_length]
        action = action.to(torch.float32).numpy()
        action = (action + 1) / 2 * (self.action_max - self.action_min + self.eps) + self.action_min
        action = np.clip(action, a_max=self.action_max, a_min=self.action_min)
        return action
    

class VelDataset(Dataset):
    def __init__(self, processor: Processor, max_time_length: int = 50, vel_chunk_length_half: int = 2, 
                    sample_per_traj: int = 4, data_downsample_ratio: int = 1):
        self.processor = processor
        self.max_time_length = max_time_length
        self.vel_chunk_length_half = vel_chunk_length_half
        self.sample_per_traj = sample_per_traj
        self.data_downsample_ratio = data_downsample_ratio
        self._load_metas()
    
    def _load_metas(self):
        dataset_statistics = self.processor.dataset_statistics
        traj_paths = dataset_statistics['traj_paths']
        self.obs_keys = dataset_statistics['obs_keys']
        self.lang_key = dataset_statistics['lang_key']
        self.action_key = dataset_statistics['action_key']
        self.proprio_key = dataset_statistics['proprio_key']
        self.control_type = dataset_statistics['control_type']
        self.metas = []
        for traj_path, traj_length in traj_paths:
            downsampled_traj_length = traj_length // self.data_downsample_ratio
            self.metas.extend([(traj_path, j, downsampled_traj_length-1) for j in range(downsampled_traj_length)])

    def _load_from_raw_traj(self, traj_path, cur_idx, goal_idx, target_idx):
        with h5py.File(traj_path, 'r') as f:
            main_view = self.obs_keys[0]
            obs_s0 = cv2.imdecode(f[main_view][cur_idx * self.data_downsample_ratio], cv2.IMREAD_COLOR)
            obs_sg = cv2.imdecode(f[main_view][goal_idx * self.data_downsample_ratio], cv2.IMREAD_COLOR)
            lang = f[self.lang_key][()]
            if isinstance(lang, (bytes, bytearray, np.bytes_)):
                lang = lang.decode("utf-8")

            vel_chunk = []
            start_idx = target_idx - self.vel_chunk_length_half
            end_idx = target_idx + self.vel_chunk_length_half
            for i in range(start_idx, end_idx + 1):
                valid_i = max(0, min(i, goal_idx))
                obs_tmp = cv2.imdecode(f[main_view][valid_i * self.data_downsample_ratio], cv2.IMREAD_COLOR)
                vel_chunk.append(obs_tmp)

        return obs_s0, obs_sg, vel_chunk, lang

    def __len__(self):
        return len(self.metas) 
    
    def __getitem__(self, index):

        traj_path, cur_idx, goal_idx = self.metas[index]
        
        left = cur_idx + 1
        right = min(cur_idx + self.max_time_length, goal_idx + max(10, self.sample_per_traj+1))
        sample_indices = random.sample(range(left, right), self.sample_per_traj-1)
        selected_target_idxs = [cur_idx] + sample_indices

        obs_s0_list = []
        obs_sg_list = []
        tau_list = []
        vel_chunk_list = []
        lang_list = []

        for target_idx in selected_target_idxs:
            tau = (target_idx - cur_idx) / self.max_time_length
            obs_s0, obs_sg, vel_chunk, lang = self._load_from_raw_traj(traj_path, cur_idx, goal_idx, target_idx)
            obs_s0, replay_params = self.processor.preprocess_image(obs_s0)
            obs_sg, _ = self.processor.preprocess_image(obs_sg, replay_params=replay_params)
            vel_chunk = [self.processor.preprocess_image(obs, replay_params=replay_params)[0] for obs in vel_chunk]
            vel_chunk = torch.stack(vel_chunk, dim=0)

            obs_s0_list.append(obs_s0)
            obs_sg_list.append(obs_sg)
            tau_list.append(torch.tensor([tau]))
            vel_chunk_list.append(vel_chunk)
            lang_list.append(lang)

        item = {
            'obs_s0': torch.stack(obs_s0_list, dim=0),
            'obs_sg': torch.stack(obs_sg_list, dim=0),
            'obs_st_chunk': torch.stack(vel_chunk_list, dim=0),
            'tau': torch.stack(tau_list, dim=0),
            'lang': lang_list
        }

        return item

def vel_collate_fn(batch):
    collated = {}

    for key in batch[0].keys():
        if key == 'lang':
            values = sum([item[key] for item in batch], [])
            collated[key] = values
        else:
            values = [item[key] for item in batch]
            collated[key] = torch.cat(values, dim=0)

    return collated

def build_vel_dataloader(meta_file_path, 
                        image_size=256,
                        sample_per_traj=4,
                        max_time_length=50, 
                        vel_chunk_length_half=2,
                        data_downsample_ratio=1,
                        batch_size=2, num_workers=2, training=True,
                        shuffle=True, pin_mem=True, drop_last=True, 
                        world_size=1, global_rank=0, **kwargs):
    
    processor = Processor(meta_file_path, image_size, training=training)
    train_dataset = VelDataset(processor=processor, max_time_length=max_time_length, data_downsample_ratio=data_downsample_ratio,
                    vel_chunk_length_half=vel_chunk_length_half, sample_per_traj=sample_per_traj)
    sampler = DistributedSampler(train_dataset, shuffle=shuffle, num_replicas=world_size, rank=global_rank) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=sampler, pin_memory=pin_mem, drop_last=drop_last, collate_fn=vel_collate_fn)
    return train_dataloader
