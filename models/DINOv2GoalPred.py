from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .DINOv2Latent import build_dino_encoder, build_dino_txt_encoder
from .DINOv2PTFlow import CrossTransBlock, get_2d_sincos_pos_embed


class Dinov2GoalPred(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="odeworld",
    repo_url="https://github.com/Dstate/ODEWorld",
    license="apache-2.0",
    tags=["robotics", "goal-prediction", "language", "dinov2"],
):
    def __init__(
        self,
        num_layers: int = 4,
        num_lang_tokens: int = 1,
        encoder_config_path: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.img_enc = build_dino_encoder(encoder_config_path=encoder_config_path)
        self.txt_enc = build_dino_txt_encoder()
        self.latent_dim = self.img_enc.latent_dim
        self.base_patches = self.img_enc.base_patches

        self.base_pos_embed = nn.Parameter(
            torch.zeros(1, self.base_patches, self.latent_dim), requires_grad=False
        )
        pos = get_2d_sincos_pos_embed(self.latent_dim, int(self.base_patches**0.5))
        self.base_pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        self.lang_proj = nn.Linear(self.txt_enc.embed_dim, self.latent_dim * num_lang_tokens)
        self.num_lang_tokens = num_lang_tokens
        self.predictor = CrossTransBlock(embed_dim=self.latent_dim, num_layers=num_layers)

        self.img_enc.requires_grad_(False)
        self.txt_enc.requires_grad_(False)

    def encode_img(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.img_enc(x)

    def encode_lang(self, lang: list[str]) -> torch.Tensor:
        with torch.no_grad():
            feat = self.txt_enc(lang)
        B = feat.shape[0]
        tokens = self.lang_proj(feat).view(B, self.num_lang_tokens, self.latent_dim)
        return tokens

    def predict_from_latents(self, s0: torch.Tensor, lang_tokens: torch.Tensor) -> torch.Tensor:
        s0_pos = s0 + self.base_pos_embed
        cond = torch.cat([s0_pos, lang_tokens], dim=1)
        return self.predictor(s0_pos, cond) - self.base_pos_embed

    def forward(self, obs_s0, obs_sg, lang, **kwargs):
        s0 = self.encode_img(obs_s0)
        sg = self.encode_img(obs_sg).detach()
        lang_tokens = self.encode_lang(lang)
        sg_pred = self.predict_from_latents(s0, lang_tokens)
        loss = F.mse_loss(sg_pred, sg)
        return loss, dict(loss=loss)

    @torch.no_grad()
    def predict(self, obs_s0, lang):
        s0 = self.encode_img(obs_s0)
        lang_tokens = self.encode_lang(lang)
        return self.predict_from_latents(s0, lang_tokens)


def build_dino_goal_pred(**kwargs):
    return Dinov2GoalPred(**kwargs)
