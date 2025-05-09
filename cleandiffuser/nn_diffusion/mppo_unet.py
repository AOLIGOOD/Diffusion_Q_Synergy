from typing import Optional, List

import torch
import torch.nn as nn

from .base_nn_diffusion import BaseNNDiffusion


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, emb_dim: int, n_heads):
        super().__init__()

        self.linear1 = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU())
        self.linear2 = nn.Sequential(nn.Linear(out_dim, out_dim), nn.SiLU())
        self.norm1 = nn.LayerNorm(out_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(out_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(out_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_dim * 6))
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # Apply linear transformations and add the condition
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        skip = self.skip(x)
        x = self.linear1(x)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = skip+gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.linear2(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MPPOUNet(BaseNNDiffusion):
    def __init__(
            self,
            act_dim: int,
            emb_dim: int = 64,
            num_heads: int = 8,
            hidden_dims: List[int] = (512, 256, 128),
            timestep_emb_type: str = "untrainable_fourier",
            timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        n_layers = len(hidden_dims)

        self.t_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        in_dim = act_dim
        for i in range(n_layers):
            self.down_blocks.append(ResidualBlock(in_dim, hidden_dims[i], emb_dim, num_heads))
            in_dim = hidden_dims[i]

        self.mid_block = ResidualBlock(in_dim, in_dim, emb_dim, num_heads)

        for i in range(n_layers - 1):
            self.up_blocks.append(
                ResidualBlock(in_dim + hidden_dims[-1 - i], hidden_dims[-2 - i], emb_dim, num_heads))
            in_dim = hidden_dims[-2 - i]

        self.out_layer = nn.Linear(in_dim, act_dim)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None, kernel_size=3):
        """
        Input:
            x:          (b, horizon, act_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, act_dim)
        """

        c = self.t_layer(self.map_noise(noise))
        if condition is not None:
            c += condition
        else:
            c += torch.zeros_like(c)

        buffer = []
        for block in self.down_blocks:
            x = block(x, c)
            buffer.append(x)

        x = self.mid_block(x, c)

        for block in self.up_blocks:
            x = torch.cat([x, buffer.pop()], dim=-1)
            x = block(x, c)

        return self.out_layer(x)
