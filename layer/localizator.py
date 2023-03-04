import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.nn.init import trunc_normal_

from config import PVT2Config
from layer.poolformer import poolformer_s12


def combine(outs: []):
    _, _, h, w = outs[0].size()
    for i in range(1, len(outs)):
        _, _, h2, w2 = outs[i].size()
        outs[i] = F.pixel_shuffle(outs[i], h // h2)
    x = torch.cat(outs, dim=1)
    return x


class Localizator(nn.Module):
    def __init__(self):
        super(Localizator, self).__init__()
        self.encode = poolformer_s12(fork_feat=True, in_chans=PVT2Config.IN_CHANNELS * 2, resblock=False)
        self.src_token = nn.Parameter(
            torch.ones(1, PVT2Config.NUM_FRAMES, PVT2Config.IN_CHANNELS, PVT2Config.IMAGE_SIZE, PVT2Config.IMAGE_SIZE))
        trunc_normal_(self.src_token, std=.02)
        self.dst_token = nn.Parameter(
            torch.ones(1, PVT2Config.NUM_FRAMES, PVT2Config.IN_CHANNELS, PVT2Config.IMAGE_SIZE,
                       PVT2Config.IMAGE_SIZE) * -1
        )
        trunc_normal_(self.dst_token, std=.02)

        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(124, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        src = x[0] + self.src_token
        fake = x[1] + self.dst_token
        x = torch.cat([src, fake], dim=2)
        x = rearrange(x, 'b t ... -> (b t) ...')
        x = self.encode(x)
        outs = combine(x)
        imgs = self.mask_head(outs)
        imgs = rearrange(imgs, '(b t) ... -> b t ...', t=PVT2Config.NUM_FRAMES)
        return imgs


if __name__ == '__main__':
    img = torch.zeros((4, PVT2Config.NUM_FRAMES, 3, 224, 224))
    net = Localizator()
    x = net([img, img])
    print(x.shape)
