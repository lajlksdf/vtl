# --------------------------------------------------------
# Pyramid Vision Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by RainbowSecret from:
#   https://github.com/whai362/PVT/blob/v2/classification/pvt_v2.py
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

from config import PVT2Config
from layer import helper
from layer.block import OverlapPatchEmbed, PVT2Block, Attention


class PVT2HashNet(nn.Module):
    def __init__(self, cfg=PVT2Config, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.cfg = cfg
        embed_dims = self.cfg.EMBED_DIMS
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, cfg.DROP_PATH_RATE, sum(cfg.DEPTHS))]
        cur = 0
        # features extractors
        for i in range(cfg.NUM_STAGES):
            patch_embed = OverlapPatchEmbed(
                img_size=cfg.IMAGE_SIZE if i == 0 else cfg.IMAGE_SIZE // (2 ** (i + 1)),
                patch_size=cfg.PATCH_SIZE if i == 0 else cfg.PATCH_SIZE // 2,
                stride=4 if i == 0 else 2,
                in_chans=cfg.IN_CHANNELS if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )

            if cfg.image_based:
                pos_embed = nn.Parameter(torch.zeros(1, patch_embed.num_patches, embed_dims[i]))
            else:
                pos_embed = nn.Parameter(torch.zeros(1, cfg.NUM_FRAMES, patch_embed.num_patches, embed_dims[i]))
            trunc_normal_(pos_embed, std=.02)

            block = nn.ModuleList([
                PVT2Block(
                    dim=embed_dims[i], num_heads=cfg.NUM_HEADS[i],
                    mlp_ratio=cfg.MLP_RATIOS[i], qkv_bias=cfg.QKV_BIAS,
                    qk_scale=cfg.QK_SCALE, drop=cfg.DROP_RATE,
                    attn_drop=cfg.ATTN_DROP_RATE, drop_path=dpr[cur + j],
                    norm_layer=norm_layer, sr_ratio=cfg.SR_RATIOS[i], linear=cfg.LINEAR,
                )
                for j in range(cfg.DEPTHS[i])
            ])
            norm = norm_layer(embed_dims[i])
            cur += cfg.DEPTHS[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self.temporal_token = nn.Parameter(torch.randn(1, PVT2Config.NUM_FRAMES, embed_dims[-1]))
        self.temporal_transformer = Attention(embed_dims[-1], embed_dims[-1], heads=PVT2Config.NUM_FRAMES)
        self.set_cls_head()
        self.set_hash_head()

        self.apply(self._init_weights)
        self.relu = nn.ReLU(inplace=True)

    def _init_weights(self, m):
        helper.init_m(m)

    def _norm_fea(self, fea):
        f = self.relu(fea)
        f = F.adaptive_avg_pool2d(f, (1, 1))
        f = f.view(f.size(0), -1)
        f = rearrange(f, '(b t) ... -> b t ...', t=self.cfg.NUM_FRAMES)
        temporal_token = repeat(self.temporal_token, '() n d -> b n d', b=f.shape[0])
        x = f + temporal_token
        x = self.temporal_transformer(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
        }  # has pos_embed may be better

    def set_cls_head(self):
        self.cls_head = nn.Sequential(
            nn.Linear(self.cfg.EMBED_DIMS[-1], self.cfg.NUM_CLASSES),
            nn.Softmax()
        )

    def set_hash_head(self):
        self.hash_head = nn.Sequential(
            nn.Linear(self.cfg.EMBED_DIMS[-1], self.cfg.HASH_BITS),
            nn.Tanh()
        )

    def forward_features(self, x):
        if not self.cfg.image_based:
            x = rearrange(x, 'b t ... -> (b t) ...')
        B = x.shape[0]
        for i in range(self.cfg.NUM_STAGES):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            if not self.cfg.image_based:
                x = rearrange(x, '(b t) ... -> b t ...', t=self.cfg.NUM_FRAMES)
            x = x + pos_embed
            if not self.cfg.image_based:
                x = rearrange(x, 'b t ... -> (b t) ...')
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self._norm_fea(x)
        x = x.mean(1)
        d, h = self.cls_head(x), self.hash_head(x)
        if self.training:
            return d, h
        else:
            return h


def load_pretrained(pretrained, hash_bits=None, cls_num=None, device=None):
    net_h = PVT2HashNet()
    params = torch.load(pretrained, map_location=device)
    for param in list(params.keys()):
        params[param.replace('feature_exact.', '')] = params.pop(param)
    net_h.load_state_dict(params)
    net_h = net_h.to(device)
    if cls_num:
        net_h.cls_head = nn.Sequential(
            nn.Linear(net_h.cfg.EMBED_DIMS[-1], cls_num),
            nn.Softmax()
        )
    if hash_bits:
        net_h.hash_head = nn.Sequential(
            nn.Linear(net_h.cfg.EMBED_DIMS[-1], hash_bits),
            nn.Tanh()
        )
    return net_h


if __name__ == '__main__':
    model = r'./512_net_h.pth'
    net_h = load_pretrained(model, cls_num=133, device=torch.device('cpu'))
    torch.save(net_h.state_dict(), '../' + helper.get_net_h())
