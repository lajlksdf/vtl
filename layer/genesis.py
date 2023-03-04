import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config import PVT2Config
from layer import helper
from layer.localizator import Localizator
from layer.vit_hash import PVT2HashNet
from util.logUtil import logger


class Genesis:
    def __init__(self, path, rank, device_ids, train_h=True, data_type=0):
        # base
        self.device = setup(rank)
        self.rank = rank
        self.data_type = data_type
        self.hash_bits = PVT2Config.HASH_BITS
        self.device_ids = device_ids
        self.train_h = train_h
        if train_h:
            self.h = PVT2HashNet()
            self.base_lr = PVT2Config.base_lr
        else:
            self.g = Localizator()
            self.base_lr = PVT2Config.base_lr * 10
        self._optimizer()
        self.init(path)

    def init(self, path):
        logger.info(f"will load:{path}")
        if self.train_h:
            path_net = path + str(self.hash_bits) + PVT2Config.NET_H
            path_json = path + str(self.hash_bits) + '_hash.json'
            self.h = self.multi_init(self.h, path_net, self.rank, self.device_ids)
            self.h.train()
            logger.info(f"loading:{path_json}")
            helper.load_hash(path_json)
        else:
            path += PVT2Config.NET_G
            self.g = self.multi_init(self.g, path, self.rank, self.device_ids)
            self.g.train()

    def _optimizer(self):
        if self.train_h:
            self.opt_h = torch.optim.Adam(self.h.parameters(), lr=self.base_lr)
        else:
            self.opt_g = torch.optim.Adam(self.g.parameters(), lr=self.base_lr)
            # self.opt_g = self.optimizer_sgd(self.g, self.base_lr)
            # self.scheduler_g = ExponentialLR(self.opt_g, gamma=0.99)

    def save(self, prefix):
        # save net
        if (PVT2Config.IS_DISTRIBUTION and self.rank == 0) or not PVT2Config.IS_DISTRIBUTION:
            if self.train_h:
                torch.save(self.h.state_dict(), prefix + helper.get_net_h(self.hash_bits))
            else:
                torch.save(self.g.state_dict(), prefix + PVT2Config.NET_G)

    def reset_grad(self):
        if self.train_h:
            self.opt_h.zero_grad()
        else:
            self.opt_g.zero_grad()

    @staticmethod
    def optimizer_sgd(m: nn.Module, learning_rate=1e-3):
        opt = torch.optim.SGD(m.parameters(), lr=learning_rate)
        return opt

    def multi_init(self, net, path, d, ids):
        if PVT2Config.IS_DISTRIBUTION:
            logger.info('MULTI_CUDA:rank:{}'.format(ids))
            net = net.to(d)
            net = DistributedDataParallel(net, device_ids=ids)
            net = net.module
            if os.path.exists(path):
                # configure map_location properly
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
                logger.info(f"loading:{path}")
                net.load_state_dict(torch.load(path, map_location=map_location))
        else:
            if os.path.exists(path):
                logger.info(f"loading:{path}")
                net.load_state_dict(torch.load(path, map_location='cuda:0'))
            net = net.to(self.device)
        return net

    def train(self):
        if self.train_h:
            self.h.train()
        else:
            self.g.train()

    def eval(self):
        if self.train_h:
            self.h.eval()
        else:
            self.g.eval()


def setup(rank):
    USE_CUDA = torch.cuda.is_available()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if USE_CUDA:
        torch.cuda.manual_seed(1)
    if torch.cuda.device_count() > 1:
        device = rank
    elif PVT2Config.IS_DISTRIBUTION:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'
        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=torch.cuda.device_count())
    else:
        device = torch.device("cuda" if USE_CUDA else "cpu")
    return device


def cleanup():
    dist.destroy_process_group()
