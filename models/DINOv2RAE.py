from typing import Optional

from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoImageProcessor
from torch import nn
import torch
import math
from diffusers.models.autoencoders.vae import Decoder
from .DINOv2Latent import build_dino_encoder


class RecDecoder(nn.Module):
    def __init__(
        self,
        dinov2_path: Optional[str] = None,
        in_channels: int = 768,
        out_channels: int = 3,
        block_out_channels = [64, 128, 256, 512, 1024],
        input_shape_is_1d = True
    ):
        super().__init__()  
        if dinov2_path is None:
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        else:
            proc = AutoImageProcessor.from_pretrained(dinov2_path)
            image_mean = proc.image_mean
            image_std = proc.image_std
        self.proc_img_mean = torch.tensor(image_mean).view(1, 3, 1, 1)
        self.proc_img_std = torch.tensor(image_std).view(1, 3, 1, 1)
        
        self.decoder = Decoder(
            in_channels=in_channels,
            out_channels=out_channels,
            up_block_types=["UpDecoderBlock2D"] * len(block_out_channels),
            block_out_channels=block_out_channels,
            mid_block_add_attention=True,
        )
        
        self.input_shape_is_1d = input_shape_is_1d

    def decode(self, z:torch.Tensor) -> torch.Tensor:
        if self.input_shape_is_1d:
            b, n, c = z.shape
            h = w = int(math.sqrt(n))
            z = z.transpose(1,2).view(b,c,h,w)
        x_rec = self.decoder(z)
        x_rec = x_rec * self.proc_img_std.to(x_rec.device) + self.proc_img_mean.to(x_rec.device)
        return x_rec
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)

class Dinov2RAE(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="odeworld",
    repo_url="https://github.com/Dstate/ODEWorld",
    license="apache-2.0",
    tags=["robotics", "image-reconstruction", "dinov2"],
):
    def __init__(
        self,
        dinov2_path: Optional[str] = None,
        noise_tau: float = 0.8,
    ):
        super().__init__()
        self.noise_tau = noise_tau
        self.encoder = build_dino_encoder(dinov2_path)
        self.latent_dim = self.encoder.latent_dim
        self.decoder = RecDecoder(dinov2_path=dinov2_path, in_channels=self.latent_dim, out_channels=3)
        self.last_layer = self.decoder.decoder.conv_out.weight

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def encode(self, x):
        return self.encoder.encode(x)

    def decode(self, z):
        return self.decoder.decode(z)

    def forward(self, obs_st):
        z = self.encoder(obs_st)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        obs_rec = self.decoder(z)
        return obs_rec


def build_dinov2rae(
        dinov2_path: Optional[str] = None,
        load_ckpt: str = None, 
        noise_tau = 0.8,
        **kwargs):
    
    model = Dinov2RAE(dinov2_path=dinov2_path, noise_tau=noise_tau)

    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(True)

    if load_ckpt:
        model.load_state_dict(torch.load(load_ckpt, map_location="cpu"), strict=True)
        model.requires_grad_(False)

    return model
