from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol
import sys
import io
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoImageProcessor, Dinov2WithRegistersConfig, Dinov2WithRegistersModel

class Stage1Protocal(Protocol):
    patch_size: int
    hidden_size: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

class Dinov2withNorm(nn.Module):

    def __init__(self, dinov2_path: Optional[str] = None, normalize: bool = True):
        super().__init__()
        if dinov2_path is None:
            config = Dinov2WithRegistersConfig(
                image_size=518,
                patch_size=14,
                interpolate_antialias=True,
                interpolate_offset=0.0,
            )
            self.encoder = Dinov2WithRegistersModel(config)
        else:
            try:
                self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
            except (OSError, ValueError, AttributeError):
                self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5
        return outputs.last_hidden_state[:, unused_token_num:]


class DinoPatchLatentEncoder(nn.Module):
    def __init__(
        self,
        encoder_cls: str = "Dinov2withNorm",
        encoder_config_path: Optional[str] = None,
        encoder_input_size: int = 224,
        encoder_params: Optional[dict] = None,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        if encoder_params is None:
            encoder_params = dict(dinov2_path=encoder_config_path, normalize=True)
        if encoder_cls != "Dinov2withNorm":
            raise ValueError(f"Unsupported encoder_cls: {encoder_cls}. Only 'Dinov2withNorm' is kept.")
        self.encoder: Stage1Protocal = Dinov2withNorm(**encoder_params)
        if encoder_config_path is None:
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        else:
            proc = AutoImageProcessor.from_pretrained(encoder_config_path)
            image_mean = proc.image_mean
            image_std = proc.image_std
        self.register_buffer("encoder_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("encoder_std", torch.tensor(image_std).view(1, 3, 1, 1))
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, (
            f"encoder_input_size {self.encoder_input_size} must be divisible by "
            f"encoder_patch_size {self.encoder_patch_size}"
        )
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2

        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")
            self.latent_mean = stats.get("mean", None)
            self.latent_var = stats.get("var", None)
            self.do_normalization = True
        else:
            self.do_normalization = False
        self.eps = eps

    def latent_encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = F.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size), mode="bicubic", align_corners=False
            )
        x = (x - self.encoder_mean) / self.encoder_std
        z = self.encoder(x)
        return z

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent_encode(x)
    

def build_dino_encoder(encoder_config_path=None):
    encoder = DinoPatchLatentEncoder(encoder_config_path=encoder_config_path)
    encoder.requires_grad_(False)
    return encoder


def _get_local_dinov2_hub_repo() -> Path:
    hub_repo = Path("~/.cache/torch/hub/facebookresearch_dinov2_main").expanduser()
    return hub_repo


def _import_dinotxt_tokenizer():
    hub_repo = _get_local_dinov2_hub_repo()

    hub_repo_str = str(hub_repo)
    if hub_repo_str not in sys.path:
        sys.path.append(hub_repo_str)

    from dinov2.hub.text.tokenizer import Tokenizer  # noqa: E402
    from dinov2.hub.utils import _DINOV2_BASE_URL  # noqa: E402

    def _build_tokenizer():
        vocab_name = "bpe_simple_vocab_16e6.txt.gz"
        tokenizer_path = hub_repo / "dinov2" / "thirdparty" / "CLIP" / "clip" / vocab_name
        if tokenizer_path.exists():
            return Tokenizer(vocab_path=str(tokenizer_path))
        else :
            if dist.is_available() and dist.is_initialized():
                if dist.get_rank() == 0:
                    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
                    url = _DINOV2_BASE_URL + f"/thirdparty/{vocab_name}"
                    resp = requests.get(url, timeout=60)
                    resp.raise_for_status()
                    tokenizer_path.write_bytes(resp.content)
                dist.barrier()
                return Tokenizer(vocab_path=str(tokenizer_path))
            else:
                tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
                url = _DINOV2_BASE_URL + f"/thirdparty/{vocab_name}"
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                tokenizer_path.write_bytes(resp.content)
                return Tokenizer(vocab_path=str(tokenizer_path))

    return _build_tokenizer

class DinoTxtLatentEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        hub_repo = _get_local_dinov2_hub_repo()
        if hub_repo.exists():
            self.model = torch.hub.load(
                str(hub_repo),
                "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
                source="local",
            ).eval()
        else:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
            ).eval()

        get_tokenizer = _import_dinotxt_tokenizer()
        self.tokenizer = get_tokenizer()
        self.embed_dim = 2048

    def latent_encode(self, x: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        text_tokens = self.tokenizer.tokenize(x).to(device)
        text_feat = self.model.encode_text(text_tokens, normalize=True)
        return text_feat

    def __call__(self, x: list[str]) -> torch.Tensor:
        return self.latent_encode(x)

def build_dino_txt_encoder():
    encoder = DinoTxtLatentEncoder()
    encoder.requires_grad_(False)
    return encoder
