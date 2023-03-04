import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from layer import helper
from layer.helper import tensor_to_binary, to_hashmap


def hash_triplet_loss(hashset, label, device):
    hashset = tensor_to_binary(hashset)
    h_map = to_hashmap(hashset, label)
    c = len(hashset) // len(label)
    intra_loss, inter_loss = 0, 0
    for k, v in h_map.items():
        for i in range(len(label)):
            l_ = label[i]
            idx = i * c
            for j in range(c):
                _loss = torch.sub(torch.from_numpy(np.asarray(v)).to(device), hashset[idx + j]).abs().mean()
                if k != l_:
                    inter_loss = (inter_loss + 1 - _loss) / 2
                if k == l_:
                    intra_loss = (intra_loss + _loss) / 2
    helper.update_hash(h_map)
    return inter_loss + intra_loss


def mask_loss(input_: Tensor, target: Tensor):
    return torch.sub(input_, target.clone().detach()).abs().mean()


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()


class Conv3d_BN(nn.Sequential):
    """
    Video Conv3d feature extraction
    Input shape:N,Cin,D,H,W; D frame size
    Output shape:N,Cout,Dout,Hout,Wout
    """

    def __init__(self, in_channels, out_channels, ks=1, stride=1, padding=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv3d(in_channels, out_channels, ks, stride, padding, bias=False))
        bn = torch.nn.BatchNorm3d(out_channels)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv3d(w.size(1), w.size(0), w.shape[2:], stride=self.c.stride, padding=self.c.padding)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16_3d(n=192, in_channels=3, activation=nn.Hardswish):
    return nn.Sequential(
        Conv3d_BN(in_channels, n // 8, 3, 2),
        activation(),
        Conv3d_BN(n // 8, n // 4, 3, 2),
        activation(),
        Conv3d_BN(n // 4, n // 2, 3, 2),
        activation(),
        Conv3d_BN(n // 2, n, 3, 2),
        nn.Dropout()
    )
