# Model definitions for LEGO
# Copyright (c) 2024, Huangjie Zheng.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
# Timestep embedding used in the DDPM++ and ADM architectures.
class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        self.positional_embedding = PositionalEmbedding(frequency_embedding_size)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.positional_embedding(t) #self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, label_dim, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Linear(label_dim, hidden_size) 
        self.label_dim = label_dim
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand([labels.shape[0],1], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, 0., labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



#################################################################################
#                                 DiT blocks                                    #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                                 LEGO Bricks                                   #
#################################################################################

class LEGO_module(nn.Module):
    """
    LEGO Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        img_resolution=32,
        patch_size=2,
        in_channels=3,
        out_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        label_dim=1000,
        learn_sigma=False,
        augment_dim=0,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.out_channels = out_channels * 2 if learn_sigma else out_channels
        in_channels = in_channels + 2 # add additional channel for coordinates.
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(img_resolution, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, c):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        class_labels: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


# dict list [hidden_size, patch_size, num_heads]
lego_module_dict = {'lego_S_2': [384, 2, 6], 'lego_S_4': [384, 4, 6], 'lego_S_8': [384, 8, 6], 'lego_S_16': [384, 16, 6], 
                    'lego_L_2': [1024, 2, 16], 'lego_L_4': [1024, 4, 16], 'lego_L_8': [1024, 8, 16], 'lego_L_16': [1024, 16, 16], 
                    'lego_XL_2': [1152, 2, 16], 'lego_XL_4': [1152, 4, 16], 'lego_XL_8': [1152, 8, 16], 'lego_XL_16': [1152, 16, 16]
                    }


class LEGO(nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        in_channels,                       # Number of color channels.
        label_dim       = 1000,                # Number of class labels, 0 = unconditional.
        class_dropout_prob=0.1,
        learn_sigma=False,
        augment_dim=0,
        model_type      = ['lego_S_2', 'lego_S_8', 'lego_S_2'],     # Class name of the underlying model.
        depths           = [2, 4, 6],               # Depth of the each brick.
        train_intermediate_stages = False,  # Return intermediate layers of the model? (Otherwise just return the final layer.)
        use_fp16 = False,                   # Use half-precision floating-point?
        **kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.max_img_size = max(img_resolution)
        self.in_channels = in_channels
        self.label_dim = label_dim
        self.num_bricks = len(img_resolution)
        model_list = [LEGO_module(img_resolution=img_resolution[0], in_channels=in_channels, out_channels=in_channels, 
                        depth=depths[0], hidden_size=lego_module_dict[model_type[0]][0], patch_size=lego_module_dict[model_type[0]][1], num_heads=lego_module_dict[model_type[0]][2], **kwargs)]
        for i in range(1, len(model_type)):
            model_list.append(LEGO_module(img_resolution=img_resolution[i], in_channels=in_channels * 2, out_channels=in_channels, 
                                depth=depths[i], hidden_size=lego_module_dict[model_type[i]][0], patch_size=lego_module_dict[model_type[i]][1], num_heads=lego_module_dict[model_type[i]][2], **kwargs))
        
        self.model = torch.nn.ModuleList(model_list)

        hidden_size = self.model[0].hidden_size
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(label_dim, hidden_size, class_dropout_prob) if label_dim else None
        self.map_augment = nn.Linear(augment_dim, hidden_size, bias=True) if augment_dim else None
        
        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize augmentat label embedding table:
        if self.map_augment is not None:
            nn.init.normal_(self.map_augment.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        self.train_intermediate_stages = train_intermediate_stages
        self.use_fp16 = use_fp16

    def convert_to_fp32(self):
        for i in range(len(self.model)):
            self.model[i].convert_to_fp32()

    def convert_to_fp16(self):
        for i in range(len(self.model)):
            self.model[i].convert_to_fp16()        

    def forward(self, x, timesteps, class_labels=None, augment_labels=None, force_drop_ids=None, return_stage_idx=None, **kwargs):
        B, C, H, W = x.shape

        # Create a grid for x and y coordinates
        y_grid, x_grid = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='xy')

        # Normalize to range [-1, 1]
        x_grid = (x_grid.float() / (W - 1) - 0.5) * 2.
        y_grid = (y_grid.float() / (H - 1) - 0.5) * 2.
        
        # Expand dims to prepare for batch size and channels
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)
        y_grid = y_grid.unsqueeze(0).unsqueeze(0)

        # Repeat across batch size and channels
        x_grid = x_grid.repeat(B, 1, 1, 1).to(x.device)
        y_grid = y_grid.repeat(B, 1, 1, 1).to(x.device)

        # Concatenate with the input tensor along the channel dimension
        x = torch.cat((x, x_grid, y_grid), dim=1) # B x (C+2) x H x W


        # proceed time-embedding, label-embedding, etc. 
        t = self.t_embedder(timesteps)                   # (B, D)
        if self.y_embedder is not None and class_labels is not None:
            y = self.y_embedder(class_labels, self.training)     # (B, D) 
            c = t + y                                # (B, D)
        else:
            c = t
        if self.map_augment is not None and augment_labels is not None:
            c = c + self.map_augment(augment_labels) 
         
        if return_stage_idx is None and self.train_intermediate_stages and (not self.training):
            return_stage_idx = len(self.model) - 1
            
        D_x = []
        for i in range(len(self.model)):
            if i == 0:
                F_x = x # B x (C + 2) x H x W
            else:
                F_x = torch.cat([F_x, x], dim=1) # B x (2*C + 2) x H x W
                
            patches = F_x.unfold(2, self.img_resolution[i], self.img_resolution[i]).unfold(3, self.img_resolution[i], self.img_resolution[i]).permute(0,2,3,1,4,5).flatten(1,2) # B x L x C x Hp x Wp (L= H*W/(Hp*Wp))

            L = patches.shape[1] # L = H*W/(Hp*Wp)
            h = w = int(math.sqrt(L)) 
            assert (h == H/self.img_resolution[i]) and (w == W/self.img_resolution[i])
            # repeat timesteps to fit in the second dimension
            c_patch = c.repeat(L, 1, 1).transpose(1,0).flatten(0,1) # B*L*D
            patches = patches.flatten(0,1) # B*L x C x Hp x Wp
            
            F_x = self.model[i](patches, c_patch)
            # unpatchify F_x
            F_x = F_x.reshape(shape=(B, h, w, C, self.img_resolution[i], self.img_resolution[i]))
            F_x = torch.einsum('nhwcpq->nchpwq', F_x) # B x h x w x C x Hp x Wp > B x C x h x Hp x w x Wp
            F_x = F_x.reshape(shape=(B, C, h * self.img_resolution[i], w * self.img_resolution[i])) #  B x C x hHp x wWp = B x C x H x W
            if (not self.training) and self.train_intermediate_stages and i == return_stage_idx:
                return F_x
            if self.train_intermediate_stages:
                D_x.append(F_x) # [B x C x H x W]
            
        if self.train_intermediate_stages and self.training:
            return D_x
        return F_x

    
    def forward_with_cfg(self, x, t, class_labels, cfg_scale, use_full_channels=True, return_stage_idx=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        x = torch.cat([x, x], 0)
        t = torch.cat([t, t], 0)
        y = class_labels
        y_null = torch.zeros_like(y)
        y = torch.cat([y, y_null], 0)
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        if use_full_channels:
            eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        else:
            eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        half_rest = rest[: len(rest) // 2]
        out = torch.cat([half_eps, half_rest], dim=1)
        return out


def lego_S_PG_64(**kwargs):
    return LEGO(img_resolution=[4, 16, 64], model_type=['lego_S_2', 'lego_S_8', 'lego_S_2'], depths=[2, 4, 6], **kwargs)

def lego_S_PR_64(**kwargs):
    return LEGO(img_resolution=[64, 16, 4], model_type=['lego_S_2', 'lego_S_8', 'lego_S_2'], depths=[6, 4, 2], **kwargs)

def lego_S_U_32(**kwargs):
    return LEGO(img_resolution=[32, 16, 4, 16, 32], model_type=['lego_S_2', 'lego_S_8', 'lego_S_2', 'lego_S_8', 'lego_S_2'], depths=[6, 4, 2, 4, 6], **kwargs)

def lego_S_U_64(**kwargs):
    return LEGO(img_resolution=[64, 16, 4, 16, 64], model_type=['lego_S_2', 'lego_S_8', 'lego_S_2', 'lego_S_8', 'lego_S_2'], depths=[6, 4, 2, 4, 6], **kwargs)

def lego_L_PG_32(**kwargs):
    return LEGO(img_resolution=[4, 16, 32], model_type=['lego_L_2', 'lego_L_8', 'lego_L_2'], depths=[4, 8, 12], **kwargs)

def lego_L_PR_32(**kwargs):
    return LEGO(img_resolution=[32, 16, 4], model_type=['lego_L_2', 'lego_L_8', 'lego_L_2'], depths=[12, 8, 4], **kwargs)

def lego_L_PG_64(**kwargs):
    return LEGO(img_resolution=[4, 16, 64], model_type=['lego_L_2', 'lego_L_8', 'lego_L_2'], depths=[6, 8, 12], **kwargs)

def lego_L_PR_64(**kwargs):
    return LEGO(img_resolution=[64, 16, 4], model_type=['lego_L_2', 'lego_L_8', 'lego_L_2'], depths=[12, 8, 6], **kwargs)

def lego_L_U_32(**kwargs):
    return LEGO(img_resolution=[32, 16, 4, 16, 32], model_type=['lego_L_2', 'lego_L_8', 'lego_L_2', 'lego_L_8', 'lego_L_2'], depths=[12, 8, 6, 8, 12], **kwargs)

def lego_L_U_64(**kwargs):
    return LEGO(img_resolution=[64, 16, 4, 16, 64], model_type=['lego_L_2', 'lego_L_8', 'lego_L_2', 'lego_L_8', 'lego_L_2'], depths=[12, 8, 6, 8, 12], **kwargs)

def lego_L_PG_64_old(**kwargs):
    return LEGO(img_resolution=[4, 16, 64], model_type=['lego_L_2', 'lego_L_8', 'lego_L_2'], depths=[6, 6, 12], **kwargs)

def lego_L_PR_64_old(**kwargs):
    return LEGO(img_resolution=[64, 16, 4], model_type=['lego_L_2', 'lego_L_8', 'lego_L_2'], depths=[12, 6, 6], **kwargs)

def lego_XL_PG_32(**kwargs):
    return LEGO(img_resolution=[4, 16, 32], model_type=['lego_XL_2', 'lego_XL_8', 'lego_XL_2'], depths=[4, 12, 14], **kwargs)

def lego_XL_PR_32(**kwargs):
    return LEGO(img_resolution=[32, 16, 4], model_type=['lego_XL_2', 'lego_XL_8', 'lego_XL_2'], depths=[14, 12, 4], **kwargs)

def lego_XL_PG_64(**kwargs):
    return LEGO(img_resolution=[4, 16, 64], model_type=['lego_XL_2', 'lego_XL_8', 'lego_XL_2'], depths=[4, 12, 14], **kwargs)

def lego_XL_PR_64(**kwargs):
    return LEGO(img_resolution=[64, 16, 4], model_type=['lego_XL_2', 'lego_XL_8', 'lego_XL_2'], depths=[14, 12, 4], **kwargs)

def lego_XL_U_32(**kwargs):
    return LEGO(img_resolution=[32, 16, 4, 16, 32], model_type=['lego_XL_2', 'lego_XL_8', 'lego_XL_2', 'lego_XL_8', 'lego_XL_2'], depths=[14, 10, 4, 10, 14], **kwargs)

def lego_XL_U_64(**kwargs):
    return LEGO(img_resolution=[64, 16, 4, 16, 64], model_type=['lego_XL_2', 'lego_XL_8', 'lego_XL_2', 'lego_XL_8', 'lego_XL_2'], depths=[14, 12, 4, 12, 14], **kwargs)

lego_models={
    'lego_S_PG_64': lego_S_PG_64, 'lego_S_PR_64': lego_S_PR_64, 'lego_S_U_32': lego_S_U_32, 'lego_S_U_64': lego_S_U_64,
    'lego_L_PG_32': lego_L_PG_32, 'lego_L_PR_32': lego_L_PR_32, 'lego_L_PG_64': lego_L_PG_64, 'lego_L_PR_64': lego_L_PR_64, 'lego_L_U_32': lego_L_U_32, 'lego_L_U_64': lego_L_U_64,
    'lego_L_PG_64_old': lego_L_PG_64_old, 'lego_L_PR_64_old': lego_L_PR_64_old, # for pkl loading with previous trained model, we recommend use lego_L_PG_64/lego_L_PR_64 for better performance
    'lego_XL_PG_32': lego_XL_PG_32, 'lego_XL_PR_32': lego_XL_PR_32, 'lego_XL_PG_64': lego_XL_PG_64, 'lego_XL_PR_64': lego_XL_PR_64, 
    'lego_XL_U_32': lego_XL_U_32, 'lego_XL_U_64': lego_XL_U_64, }


