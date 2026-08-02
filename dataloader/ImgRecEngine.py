import os
import os.path as osp
import torch
import json 
import numpy as np
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, BatchSampler, DistributedSampler
import h5py
import cv2
import torch
from typing import Any, Dict, Union
import traceback
import uvicorn
import json_numpy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI
from fastapi.responses import JSONResponse
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


def build_base_transform(n_px, aug=True, crop_scale=0.95, crop_prob=0.9):

    base_transform = []
    base_transform.append(A.Resize(height=n_px, width=n_px))
    base_transform.append(A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)))
    base_transform.append(ToTensorV2())
    base_transform = A.ReplayCompose(base_transform)
    return base_transform


class Processor(object):
    def __init__(self, meta_file_path, image_size=256, training=True, eps=1e-6):
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
    

class UniDataset(Dataset):
    def __init__(self, processor: Processor):
        self.processor = processor
        self.chunk_length = 16
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
        for traj_path, traj_lengh in traj_paths:
            self.metas.extend([(traj_path, j, traj_lengh-1) for j in range(traj_lengh)])

    def _load_from_raw_traj(self, traj_path, cur_idx, goal_idx):
        with h5py.File(traj_path, 'r') as f:
            main_view = self.obs_keys[0]
            obs_st = cv2.imdecode(f[main_view][cur_idx], cv2.IMREAD_COLOR)

            return obs_st

    def __len__(self):
        return len(self.metas) 
    
    def __getitem__(self, index):

        meta = self.metas[index]
        obs_st = self._load_from_raw_traj(meta[0], meta[1], meta[2])
        obs_st, replay_params = self.processor.preprocess_image(obs_st)

        item = {
            'obs': obs_st,
        }

        return item

def build_uni_dataloader(meta_file_path, image_size=256, 
                        training=True, batch_size=2, num_workers=2, 
                        shuffle=True, pin_mem=True, drop_last=True, 
                        world_size=1, global_rank=0, **kwargs):
    
    processor = Processor(meta_file_path, image_size, training=training)
    train_dataset = UniDataset(processor=processor)
    sampler = DistributedSampler(train_dataset, shuffle=shuffle, num_replicas=world_size, rank=global_rank) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=sampler, pin_memory=pin_mem, drop_last=drop_last)
    return train_dataloader
