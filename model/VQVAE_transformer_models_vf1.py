# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.nn import functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PointEmbedder(nn.Module):
    def __init__(self, hidden_size, out_size, bias=True):
        super().__init__()
        self.proj = nn.Linear(hidden_size, out_size, bias=bias)

    def forward(self, p):
        p = self.proj(p)
        return p

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        x = self.tanh(x)
        return x


class VQVAE_Transformer(nn.Module):
    def __init__(
        self,
        input_size=32,
        in_channels=4,
        hidden_size=1152,
        num_cbembed=512,
        dim_cbembed=64,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        beta=0.25
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        #####//#####
        self.p_embedder = PointEmbedder(in_channels, hidden_size, bias=True)
        self.pos_embed = nn.Parameter(torch.randn(1, input_size, hidden_size))

        self.encoder_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.encod_emb = nn.Linear(hidden_size, dim_cbembed)
        self.decod_emb = nn.Linear(dim_cbembed, hidden_size)

        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

        self.vq_layer = VectorQuantizer(num_cbembed, dim_cbembed, beta)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x):
        # Encoder
        x = self.p_embedder(x) + self.pos_embed
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = self.encod_emb(x)

        # VQ layer
        quantized_inputs, vq_loss = self.vq_layer(x)
        quantized_inputs = self.decod_emb(quantized_inputs)
        
        # Decoder
        z = quantized_inputs + self.pos_embed
        for decoder_block in self.decoder_blocks:
            z = decoder_block(z)
        recon = self.final_layer(z)

        return recon, vq_loss, x


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def VQVAET_XL(**kwargs):
    return VQVAE_Transformer(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def VQVAET_L(**kwargs):
    return VQVAE_Transformer(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def VQVAET_B(**kwargs):
    return VQVAE_Transformer(depth=12, hidden_size=768, num_heads=12, **kwargs)

def VQVAET_S(**kwargs):
    return VQVAE_Transformer(depth=12, hidden_size=384, num_heads=6, **kwargs)

def VQVAET_T(**kwargs):
    return VQVAE_Transformer(depth=12, hidden_size=192, num_heads=3, **kwargs)

VQVAE_Transformer_models = {
    'VQVAET-XL': VQVAET_XL,  
    'VQVAET-L':  VQVAET_L,
    'VQVAET-B':  VQVAET_B,
    'VQVAET-S':  VQVAET_S,
    'VQVAET-T':  VQVAET_T
}

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        #latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x N x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents, vq_loss  # [B x N x D]