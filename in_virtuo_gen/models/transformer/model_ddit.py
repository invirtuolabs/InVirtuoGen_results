# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/facebookresearch/DiT
# which is released under NonCommercial-4.0 license
# Part of this implementation is adapted from https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
# which is released under MIT license
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from . import rotary



def bias_dropout_add_scale(x: Tensor, scale: Tensor, residual: Optional[Tensor], prob: float, training: bool) -> Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x = F.layer_norm(x.float(), [self.dim])

        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(time)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_conds: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be devisable by n_heads"

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout

        self.head_dim = self.dim // self.n_heads

        self.norm1 = LayerNorm(dim=dim)

        self.qw = nn.Linear(dim, dim, bias=False)
        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)

        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        self.adaLN_modulation = nn.Linear(n_conds, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, rotary_cos_sin: Tensor, c: Tensor, attn_mask: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c)[
            :, None
        ].chunk(6, dim=2)

        x_skip = x
        x = modulate(x=self.norm1(x), shift=shift_msa, scale=scale_msa)

        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        q, k, v = (item.view(batch_size, seq_len, self.n_heads, self.head_dim) for item in (q, k, v))

        # with torch.amp.autocast("cuda", enabled=False):
        cos, sin = rotary_cos_sin
        original_dtype = q.dtype

        q = rotary.apply_rotary_emb_torch(x=q, cos=cos, sin=sin).to(original_dtype)
        k = rotary.apply_rotary_emb_torch(x=k, cos=cos, sin=sin).to(original_dtype)

        q, k, v = (item.transpose(1, 2) for item in (q, k, v))

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_mask)
        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)
        x = bias_dropout_add_scale(
            x=self.attn_out(x),
            scale=gate_msa,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )
        x = bias_dropout_add_scale(
            x=self.mlp(modulate(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
            scale=gate_mlp,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )

        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, n_conds: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(n_conds, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)

        return x


class DDiT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        t_emb_dim: int = 128,
        n_heads: int = 12,
        n_layer: int = 12,
        dropout: float = 0.1,
        classification: bool = False,
        num_classes: int = 1,
        n_conds: int = 0,
    ):
        super().__init__()
        self.classification = classification
        self.vocab_size = vocab_size

        self.vocab_embed = nn.Embedding(self.vocab_size, hidden_size)

        self.time_embedding = TimestepEmbedder(hidden_size=t_emb_dim)
        self.rotary_emb = rotary.Rotary(dim=hidden_size // n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=hidden_size,
                    n_heads=n_heads,
                    n_conds=t_emb_dim,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )

        self.output_layer = DDitFinalLayer(
            hidden_size=hidden_size,
            out_channels=vocab_size,
            n_conds=t_emb_dim,
        )
        if classification:
            # Create a learnable classification token.
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
            # self.cls_out = nn.Linear(hidden_size, num_classes)
        if n_conds > 0:
            self.conds = nn.Sequential(
                nn.Linear(n_conds, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, t_emb_dim),
            )
        else:
            self.conds = None

    def forward(self, x: Tensor, t: Tensor, attn_mask: Tensor, conds: Optional[Tensor] = None, return_hidden=False) -> Tensor:
        # attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        # with torch.amp.autocast("cuda", enabled=False):
        # Embed the input token IDs
        x = self.vocab_embed(x)  # Shape: (batch_size, seq_len, hidden_size)

        if self.classification:
            # Prepend the classification token to the sequence.
            batch_size = x.size(0)
            attn_mask = attn_mask[:, :, :]
            # cls_tokens = self.cls_token.expand(batch_size, 1, -1)  # Shape: (batch_size, 1, hidden_size)
            # # x = torch.cat([ x,cls_tokens], dim=1)  # New shape: (batch_size, seq_len+1, hidden_size)
            # x = torch.cat([cls_tokens, x], dim=1)  # New shape: (batch_size, seq_len+1, hidden_size)

        # Compute conditioning vector from the time embedding
        c = F.silu(self.time_embedding(time=t))
        if conds is not None:
            c += self.conds(conds)
        else:
            assert self.conds is None

        # Get rotary embeddings
        rotary_cos_sin = self.rotary_emb(x=x)

        for block in self.blocks:
            x = block(x=x, rotary_cos_sin=rotary_cos_sin, c=c, attn_mask=attn_mask)

        if self.classification:
            return x
        else:
            logits = self.output_layer(x=x, c=c)
            if not return_hidden:
                return logits
            else:
                return logits,x
