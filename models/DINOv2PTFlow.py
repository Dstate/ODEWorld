from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import jvp
from huggingface_hub import PyTorchModelHubMixin
from torchdiffeq import odeint
from .DINOv2Latent import build_dino_encoder, build_dino_txt_encoder

class LearnedPosEmb(nn.Module):
    def __init__(self, input_size, output_size, init_scale = 0.2):
        super().__init__()
        assert output_size % 2 == 0
        self.output_size = output_size
        self.kernel = nn.Parameter(torch.randn(output_size // 2, input_size) * init_scale)

    def forward(self, x):
        f = torch.pi * x @ self.kernel.T
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length):
    lis = np.arange(length, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, lis)
    return pos_embed

def estimate_velocity_fir(x):
    Ws = x.shape[1]
    assert Ws % 2 == 1
    center = Ws // 2

    t = torch.arange(-center, center + 1, device=x.device, dtype=x.dtype)
    kernel = t / (t.pow(2).sum() + 1e-8)
    kernel_shape = [1, Ws] + [1] * (x.dim() - 2)
    kernel = kernel.view(*kernel_shape)
    vel = (x * kernel).sum(dim=1)
    return vel

def noising(x, std=0.1):
    return x + torch.randn_like(x) * std

@torch.no_grad()
def decode_rollout_chunked(decode_fn, s0, z_traj, chunk_size=64):
    """Decode a rollout [B, T, num_tokens, D] chunk-by-chunk along T.

    Feeding all T frames as one B*T batch through the decoder spikes peak
    activation memory (OOM for long rollouts); chunking bounds the peak.
    """
    B, T = z_traj.shape[0], z_traj.shape[1]
    D = s0.shape[-1]
    outs = []
    for i in range(0, T, chunk_size):
        z_chunk = z_traj[:, i : i + chunk_size]
        t = z_chunk.shape[1]
        s0_rep = s0.unsqueeze(1).expand(B, t, -1, D).reshape(B * t, -1, D)
        rec = decode_fn(s0_rep, z_chunk.reshape(B * t, -1, D))
        outs.append(rec.reshape(B, t, rec.shape[1], rec.shape[2]))
    return torch.cat(outs, dim=1)

class FilmMlp(nn.Module):
    def __init__(self, input_dim, cond_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.main_net = nn.ModuleList()
        self.main_net.append(nn.Linear(input_dim, hidden_dim))
        self.main_net.append(nn.ReLU())
        for _ in range(num_layers - 2):
            self.main_net.append(nn.Linear(hidden_dim, hidden_dim))
            self.main_net.append(nn.ReLU())
        self.main_net.append(nn.Linear(hidden_dim, output_dim))
        self.film_gen = nn.Linear(cond_dim, hidden_dim * 2 * (num_layers - 1))
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

    def forward(self, x, cond):
        gamma_beta = self.film_gen(cond)
        idx = 0
        for i, layer in enumerate(self.main_net):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                if idx + 2 * x.shape[-1] > gamma_beta.shape[-1]:
                    break
                g = 1.0 + gamma_beta[:, idx : idx + x.shape[-1]]
                b = gamma_beta[:, idx + x.shape[-1] : idx + 2 * x.shape[-1]]
                idx += 2 * x.shape[-1]

                g = g.unsqueeze(1)
                b = b.unsqueeze(1)
                x = x * g + b
        return x


class CrossAttnLayer(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        dim_feedforward=4096,
        num_heads=12,
        activation=nn.GELU(),
        drop_out_rate=0.0,
    ):
        super(CrossAttnLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = activation
        self.dropout1 = nn.Dropout(drop_out_rate)
        self.dropout2 = nn.Dropout(drop_out_rate)
        self.dropout3 = nn.Dropout(drop_out_rate)

    def forward(self, x, cond):
        attn_output, attn_weights = self.attn(x, cond, cond)
        x = self.norm1(x + self.dropout1(attn_output))
        x = self.norm2(x + self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x))))))
        return x, attn_weights


class CrossTransBlock(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        dim_feedforward=2048,
        num_heads=8,
        num_layers=3,
        activation=nn.GELU(),
        drop_out_rate=0.0,
    ):
        super(CrossTransBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.activation = activation
        self.drop_out_rate = drop_out_rate

        self.layers = nn.ModuleList(
            [
                CrossAttnLayer(embed_dim, dim_feedforward, num_heads, activation, drop_out_rate)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, cond, return_attn_weights=False):
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, cond)
            all_attn_weights.append(attn_weights)

        if return_attn_weights:
            return x, all_attn_weights
        else:
            return x

class Dinov2PTflowImgoal(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="odeworld",
    repo_url="https://github.com/Dstate/ODEWorld",
    license="apache-2.0",
    tags=["robotics", "world-model", "continuous-time", "dinov2"],
):
    def __init__(
        self,
        rec_weight: float = 1.0,
        dyn_enc_weight: float = 1.0,
        num_delta_tokens=1,
        v_hidden_dim: int = 4096,
        v_num_layers: int = 3,
        max_time_length: int = 200,
        disable_encoder_detach: bool = False,
        encoder_config_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = build_dino_encoder(encoder_config_path=encoder_config_path)
        self.latent_dim = self.backbone.latent_dim
        self.base_patches = self.backbone.base_patches

        self.base_pos_embed = nn.Parameter(torch.zeros(1, self.base_patches, self.latent_dim), requires_grad=False)
        tmp_pos_embed = get_2d_sincos_pos_embed(self.latent_dim, int(self.base_patches**0.5))
        self.base_pos_embed.data.copy_(torch.from_numpy(tmp_pos_embed).float().unsqueeze(0))

        self.delta_trans_blocks = CrossTransBlock(embed_dim=self.latent_dim, num_layers=4)
        self.rec_trans_blocks = CrossTransBlock(embed_dim=self.latent_dim, num_layers=4)
        self.delta_token = nn.Parameter(torch.randn(1, num_delta_tokens, self.latent_dim))
        nn.init.trunc_normal_(self.delta_token, std=0.01)
        self.num_delta_tokens = num_delta_tokens

        self.t_dim = self.latent_dim
        self.t_process = LearnedPosEmb(1, self.t_dim)
        
        self.v_model = FilmMlp(
            input_dim=self.latent_dim * 3,           
            cond_dim=self.t_dim,                
            output_dim=self.latent_dim,
            hidden_dim=v_hidden_dim,
            num_layers=v_num_layers
        )
        self.rec_weight = rec_weight
        self.dyn_enc_weight = dyn_enc_weight
        self.max_time_length = max_time_length
        self.disable_encoder_detach = disable_encoder_detach

    def latent_encode(self, x: torch.Tensor) -> torch.Tensor:
        s = self.backbone(x)
        return s
    
    def delta_decouple(self, s0, st):
        B, N, C = st.shape

        q_tokens = self.delta_token.expand(B, -1, -1)
        kv_tokens =  torch.cat([s0 + self.base_pos_embed, st + self.base_pos_embed], dim=1)
        z_trans = self.delta_trans_blocks(q_tokens, kv_tokens)

        return z_trans
    
    def delta_decode(self, s0, z_trans):
        rec_st =  self.rec_trans_blocks(s0 + self.base_pos_embed, z_trans)
        rec_st = rec_st - self.base_pos_embed 
        return rec_st

    def forward_vmodel(self, z0, ztau, zg, tau):
        z_in = torch.cat([z0, ztau, zg], dim=-1)
        if tau.ndim < 2:
            tau = tau.reshape(-1, 1).expand(ztau.shape[0], 1)
        t_emb = self.t_process(tau)
        v_pred = self.v_model(z_in, t_emb) 
        
        return v_pred

    def forward(self, obs_s0, obs_sg, obs_st_chunk, tau, rec_warm = False, **kwargs):
        B, T, C, H, W = obs_st_chunk.shape
        st_chunk = self.latent_encode(obs_st_chunk.reshape(B*T, C, H, W))
        _, N, D = st_chunk.shape
        st_chunk = st_chunk.reshape(B, T, N, D)
        ds_dt = estimate_velocity_fir(st_chunk) 
        ds_dtau = ds_dt

        s0 = self.latent_encode(obs_s0)
        sg = self.latent_encode(obs_sg)
        stau = st_chunk[:, T // 2, ...]

        z0 = self.delta_decouple(s0, s0)
        zg = self.delta_decouple(s0, sg)
        ztau, target_dz_dtau = jvp(self.delta_decouple, (s0, stau), (torch.zeros_like(s0), ds_dtau))
        pred_v = self.forward_vmodel(z0, ztau, zg, tau)

        rec_stau = self.delta_decode(s0, ztau)

        if self.disable_encoder_detach:
            loss_dyn_enc = F.mse_loss(pred_v, target_dz_dtau, reduction='none').mean()
        else:
            loss_dyn_enc = F.mse_loss(pred_v, target_dz_dtau.detach(), reduction='none').mean()
        loss_rec = F.mse_loss(stau, rec_stau)

        if rec_warm:
            loss = self.rec_weight * loss_rec + 0 * loss_dyn_enc
            return loss, dict(
                loss=loss, 
                loss_rec=loss_rec, 
                loss_dyn_enc=loss_dyn_enc,
                stau_mean=stau.mean(),
                stau_std=stau.std(),
                ds_dtau_mean=ds_dtau.mean(),
                ds_dtau_std=ds_dtau.std(),
            )
        else :
            loss = self.rec_weight * loss_rec + self.dyn_enc_weight * loss_dyn_enc
            return loss, dict(
                loss=loss, 
                loss_rec=loss_rec, 
                loss_dyn_enc=loss_dyn_enc,
                stau_mean=stau.mean(),
                stau_std=stau.std(),
                ds_dtau_mean=ds_dtau.mean(),
                ds_dtau_std=ds_dtau.std(),
            )

    @torch.no_grad()
    def rollout_ode_lang(self, obs_s0, lang, goal_predictor, horizon=0.5, steps=25, ode_method="rk4"):
        B, device = obs_s0.shape[0], obs_s0.device
        s0 = self.latent_encode(obs_s0)
        sg = goal_predictor.predict(obs_s0, lang)
        z0 = self.delta_decouple(s0, s0)
        zg = self.delta_decouple(s0, sg)

        def ode_func(t_scalar, z):
            tau = t_scalar.view(1, 1).expand(B, 1)
            return self.forward_vmodel(z0, z, zg, tau) * self.max_time_length

        t_grid = torch.linspace(0, horizon, steps + 1, device=device)
        rollouts = odeint(ode_func, z0, t_grid, method=ode_method)[1:]
        rollouts = rollouts.permute(1, 0, 2, 3)
    
        rec_seq = decode_rollout_chunked(self.delta_decode, s0, rollouts)
        return rec_seq, rollouts

    @torch.no_grad()
    def rollout_ode(self, obs_s0, obs_sg, horizon=0.5, steps=25, ode_method="rk4"):
        B, device = obs_s0.shape[0], obs_s0.device
        s0 = self.latent_encode(obs_s0)
        sg = self.latent_encode(obs_sg)
        z0 = self.delta_decouple(s0, s0)
        zg = self.delta_decouple(s0, sg)

        def ode_func(t_scalar, z):
            tau = t_scalar.view(1, 1).expand(B, 1)
            return self.forward_vmodel(z0, z, zg, tau) * self.max_time_length

        t_grid = torch.linspace(0, horizon, steps + 1, device=device)
        rollouts = odeint(ode_func, z0, t_grid, method=ode_method)[1:]
        rollouts = rollouts.permute(1, 0, 2, 3)
    
        rec_seq = decode_rollout_chunked(self.delta_decode, s0, rollouts)
        return rec_seq, rollouts
    
    @torch.no_grad()
    def rollout_ode_replan(self, obs_s0, obs_sg, horizon=0.5, steps=25, replan_steps=5, overlap=5, ode_method="rk4"):
        B = obs_s0.shape[0]
        device = obs_s0.device
        current_s = self.latent_encode(obs_s0)
        sg = self.latent_encode(obs_sg)
        all_s = []
        prev_chunk = None
        for i in range(replan_steps):
            z0 = self.delta_decouple(current_s, current_s)
            zg = self.delta_decouple(current_s, sg)

            def ode_func(t_scalar, z):
                tau = t_scalar.view(1, 1).expand(B, 1)
                return self.forward_vmodel(z0, z, zg, tau) * self.max_time_length
                
            t_grid = torch.linspace(0, horizon, steps + 1, device=device)
            z_traj = odeint(ode_func, z0, t_grid, method=ode_method)[1:]
            z_traj = z_traj.permute(1, 0, 2, 3)

            chunk_s = decode_rollout_chunked(self.delta_decode, current_s, z_traj)

            if prev_chunk is not None:
                k = min(overlap, prev_chunk.shape[1], chunk_s.shape[1])
                if k > 0:
                    old_part = prev_chunk[:, -k:]
                    new_part = chunk_s[:, :k]
                    weights = torch.linspace(0, 1, k, device=device).view(1, k, 1, 1)
                    blended = (1 - weights) * old_part + weights * new_part
                    all_s[-1] = all_s[-1][:, :-k]
                    chunk_s = torch.cat([blended, chunk_s[:, k:]], dim=1)

            all_s.append(chunk_s)
            prev_chunk = chunk_s
            current_s = chunk_s[:, -1]

        all_s = torch.cat(all_s, dim=1)
        return all_s


class Dinov2PTflowLangoal(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="odeworld",
    repo_url="https://github.com/Dstate/ODEWorld",
    license="apache-2.0",
    tags=["robotics", "world-model", "continuous-time", "dinov2"],
):
    def __init__(
        self,
        rec_weight: float = 1.0,
        dyn_enc_weight: float = 1.0,
        num_delta_tokens=1,
        v_hidden_dim: int = 4096,
        v_num_layers: int = 3,
        max_time_length: int = 200,
        disable_encoder_detach: bool = False,
        encoder_config_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = build_dino_encoder(encoder_config_path=encoder_config_path)
        self.text_encoder = build_dino_txt_encoder()
        self.latent_dim = self.backbone.latent_dim
        self.base_patches = self.backbone.base_patches

        self.base_pos_embed = nn.Parameter(torch.zeros(1, self.base_patches, self.latent_dim), requires_grad=False)
        tmp_pos_embed = get_2d_sincos_pos_embed(self.latent_dim, int(self.base_patches**0.5))
        self.base_pos_embed.data.copy_(torch.from_numpy(tmp_pos_embed).float().unsqueeze(0))

        self.delta_trans_blocks = CrossTransBlock(embed_dim=self.latent_dim, num_layers=4)
        self.rec_trans_blocks = CrossTransBlock(embed_dim=self.latent_dim, num_layers=4)
        self.delta_token = nn.Parameter(torch.randn(1, num_delta_tokens, self.latent_dim))
        nn.init.trunc_normal_(self.delta_token, std=0.01)
        self.num_delta_tokens = num_delta_tokens

        self.lang_proj = nn.Linear(self.text_encoder.embed_dim, self.latent_dim * num_delta_tokens)

        self.t_dim = self.latent_dim
        self.t_process = LearnedPosEmb(1, self.t_dim)
        
        self.v_model = FilmMlp(
            input_dim=self.latent_dim * 3,           
            cond_dim=self.t_dim,                
            output_dim=self.latent_dim,
            hidden_dim=v_hidden_dim,
            num_layers=v_num_layers
        )
        self.rec_weight = rec_weight
        self.dyn_enc_weight = dyn_enc_weight
        self.max_time_length = max_time_length
        self.disable_encoder_detach = disable_encoder_detach

    def latent_encode(self, x: torch.Tensor) -> torch.Tensor:
        s = self.backbone(x)
        return s

    def encode_lang(self, lang: list[str]) -> torch.Tensor:
        with torch.no_grad():
            feat = self.text_encoder(lang)
        B = feat.shape[0]
        return self.lang_proj(feat).view(B, self.num_delta_tokens, self.latent_dim)
    
    def delta_decouple(self, s0, st):
        B, N, C = st.shape

        q_tokens = self.delta_token.expand(B, -1, -1)
        kv_tokens =  torch.cat([s0 + self.base_pos_embed, st + self.base_pos_embed], dim=1)
        z_trans = self.delta_trans_blocks(q_tokens, kv_tokens)

        return z_trans
    
    def delta_decode(self, s0, z_trans):
        rec_st =  self.rec_trans_blocks(s0 + self.base_pos_embed, z_trans)
        rec_st = rec_st - self.base_pos_embed 
        return rec_st

    def forward_vmodel(self, z0, ztau, zg, tau):
        z_in = torch.cat([z0, ztau, zg], dim=-1)
        if tau.ndim < 2:
            tau = tau.reshape(-1, 1).expand(ztau.shape[0], 1)
        t_emb = self.t_process(tau)
        v_pred = self.v_model(z_in, t_emb) 
        
        return v_pred

    def forward(self, obs_s0, obs_st_chunk, tau, lang, rec_warm=False, **kwargs):
        B, T, C, H, W = obs_st_chunk.shape
        st_chunk = self.latent_encode(obs_st_chunk.reshape(B*T, C, H, W))
        _, N, D = st_chunk.shape
        st_chunk = st_chunk.reshape(B, T, N, D)
        ds_dt = estimate_velocity_fir(st_chunk) 
        ds_dtau = ds_dt

        s0 = self.latent_encode(obs_s0)
        stau = st_chunk[:, T // 2, ...]

        z0 = self.delta_decouple(s0, s0)
        zg = self.encode_lang(lang)
        ztau, target_dz_dtau = jvp(self.delta_decouple, (s0, stau), (torch.zeros_like(s0), ds_dtau))
        pred_v = self.forward_vmodel(z0, ztau, zg, tau)

        rec_stau = self.delta_decode(s0, ztau)

        if self.disable_encoder_detach:
            loss_dyn_enc = F.mse_loss(pred_v, target_dz_dtau, reduction='none').mean()
        else:
            loss_dyn_enc = F.mse_loss(pred_v, target_dz_dtau.detach(), reduction='none').mean()
        loss_rec = F.mse_loss(stau, rec_stau)

        if rec_warm:
            loss = self.rec_weight * loss_rec + 0 * loss_dyn_enc
            return loss, dict(
                loss=loss, 
                loss_rec=loss_rec, 
                loss_dyn_enc=loss_dyn_enc,
                stau_mean=stau.mean(),
                stau_std=stau.std(),
                ds_dtau_mean=ds_dtau.mean(),
                ds_dtau_std=ds_dtau.std(),
            )
        else :
            loss = self.rec_weight * loss_rec + self.dyn_enc_weight * loss_dyn_enc
            return loss, dict(
                loss=loss, 
                loss_rec=loss_rec, 
                loss_dyn_enc=loss_dyn_enc,
                stau_mean=stau.mean(),
                stau_std=stau.std(),
                ds_dtau_mean=ds_dtau.mean(),
                ds_dtau_std=ds_dtau.std(),
            )

    @torch.no_grad()
    def rollout_ode(self, obs_s0, lang, horizon=0.5, steps=25, ode_method="rk4"):
        B, device = obs_s0.shape[0], obs_s0.device
        s0 = self.latent_encode(obs_s0)
        z0 = self.delta_decouple(s0, s0)
        zg = self.encode_lang(lang)

        def ode_func(t_scalar, z):
            tau = t_scalar.view(1, 1).expand(B, 1)
            return self.forward_vmodel(z0, z, zg, tau) * self.max_time_length

        t_grid = torch.linspace(0, horizon, steps + 1, device=device)
        rollouts = odeint(ode_func, z0, t_grid, method=ode_method)[1:]
        rollouts = rollouts.permute(1, 0, 2, 3)

        rec_seq = decode_rollout_chunked(self.delta_decode, s0, rollouts)
        return rec_seq, rollouts

    @torch.no_grad()
    def rollout_ode_replan(self, obs_s0, lang, horizon=0.5, steps=25, replan_steps=5, overlap=5, ode_method="rk4"):
        B = obs_s0.shape[0]
        device = obs_s0.device
        current_s = self.latent_encode(obs_s0)
        zg = self.encode_lang(lang)
        all_s = []
        prev_chunk = None
        for i in range(replan_steps):
            z0 = self.delta_decouple(current_s, current_s)

            def ode_func(t_scalar, z):
                tau = t_scalar.view(1, 1).expand(B, 1)
                return self.forward_vmodel(z0, z, zg, tau) * self.max_time_length

            t_grid = torch.linspace(0, horizon, steps + 1, device=device)
            z_traj = odeint(ode_func, z0, t_grid, method=ode_method)[1:]
            z_traj = z_traj.permute(1, 0, 2, 3)

            chunk_s = decode_rollout_chunked(self.delta_decode, current_s, z_traj)

            if prev_chunk is not None:
                k = min(overlap, prev_chunk.shape[1], chunk_s.shape[1])
                if k > 0:
                    old_part = prev_chunk[:, -k:]
                    new_part = chunk_s[:, :k]
                    weights = torch.linspace(0, 1, k, device=device).view(1, k, 1, 1)
                    blended = (1 - weights) * old_part + weights * new_part
                    all_s[-1] = all_s[-1][:, :-k]
                    chunk_s = torch.cat([blended, chunk_s[:, k:]], dim=1)

            all_s.append(chunk_s)
            prev_chunk = chunk_s
            current_s = chunk_s[:, -1]

        all_s = torch.cat(all_s, dim=1)
        return all_s
