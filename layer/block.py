import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum
from torch.nn import Dropout

from layer import helper


class SRAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1,
                 linear=False, ):
        super().__init__()
        assert (dim % num_heads == 0), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        helper.init_m(m)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = (self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
            else:
                kv = (self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = (self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3, hw=14, padding=1):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(scale_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=hw, w=hw)
                                     )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=padding, groups=scale_dim, bias=False),
            nn.MaxPool2d(kernel_size=depth_kernel, stride=1, padding=padding),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=hw, w=hw)
        )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class UpperSample(nn.Module):

    def __init__(self, in_dim, out_dim, hw):
        super(UpperSample, self).__init__()

        self.up_proj = nn.Sequential(nn.Linear(in_dim, in_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(in_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=hw, w=hw)
                                     )

        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', c=out_dim)
        )

        self.to_out = nn.Sequential(nn.Linear(out_dim, out_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(out_dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c'),
                                    nn.Dropout()
                                    )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.de_conv(x)
        x = self.to_out(x)
        return x


class LCAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ReAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rate=2, norm=nn.LayerNorm):
        super(LinearBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels * rate, bias=False),
            nn.GELU(),
            nn.Linear(in_channels * rate, out_channels, bias=False),
            norm(out_channels)
        )

    def forward(self, x):
        return self.fc(x)


class LinearBn(nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.fc = nn.Linear(a, b, bias=False)
        self.bn = nn.BatchNorm1d(b)
        nn.init.constant_(self.bn.weight, bn_weight_init)

    def forward(self, x):
        x = self.fc(x)
        return self.bn(x.flatten(0, 1)).reshape_as(x)


class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
            LinearBn(dim, out_dim)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


def b16(n, activation=nn.GELU, in_channels=3):
    return torch.nn.Sequential(
        Conv2d_BN(in_channels, n // 8, 3, 2, 1),
        activation(),
        ResnetBlock(n // 8, n // 4),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1),
        activation(),
        ResnetBlock(n // 2, n))


def conv_blocks(n, activation=nn.GELU, in_channels=3):
    return torch.nn.Sequential(
        ResnetBlock(in_channels, n // 8),
        activation(),
        ResnetBlock(n // 8, n // 4),
        activation(),
        ResnetBlock(n // 4, n // 2),
        activation(),
        ResnetBlock(n // 2, n))


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size // patch_size, img_size // patch_size
        self.num_patches = ((img_size - patch_size + (patch_size // 2) * 2) // stride + 1) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2, )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        helper.init_m(m)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PVT2Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False, ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SRAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                proj_drop=drop, sr_ratio=sr_ratio, linear=linear, )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = Dropout(p=drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear,
                       down_sample=sr_ratio != 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        helper.init_m(m)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, linear=False,
                 down_sample=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.down_sample = down_sample
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        helper.init_m(m)

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        if self.down_sample:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ResnetBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, down=False):
        super(ResnetBlock, self).__init__()
        self.down = down
        self.conv1 = Conv2d_BN(inplanes, inplanes * self.expansion)
        self.conv2 = Conv2d_BN(inplanes * self.expansion, inplanes * self.expansion, 3, 1, 1)
        self.conv3 = Conv2d_BN(inplanes * self.expansion, inplanes)
        self.stride = stride
        if down:
            self.up = Conv2d_BN(inplanes, planes, 3, 2, 1)
        else:
            self.up = Conv2d_BN(inplanes, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual

        return self.up(out)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.c = torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(b, momentum=0.1)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.c(x)
        x = self.bn(x)
        return x
