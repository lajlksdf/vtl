import _thread
import argparse
import time

import torch.nn as nn

from config import PVT2Config
from dataset.BaseDataset import TrainCache, load_cache, get_dataloader, TrainItem
from dataset.DFD_Video import DFDVideoDataset
from dataset.DFTL import DFTLDataset
from dataset.FFTL import FFTLDataset
from dataset.inpainting_dataset import InpaintingDataset
from dataset.splicingtl import VSTLDataset
from layer import fn
from layer.genesis import Genesis
from layer.helper import cb2b
from util import figureUtil
from util.logUtil import logger

mask_loss = nn.MSELoss()

# program init
bce_loss = nn.BCELoss(reduction='mean')
itr_times, g_losses, h_losses, d_losses, h_d_losses, accuracies, hash_dists = [], [], [], [], [], [], []
choices = {  # (number of original videos, Dataset: read frames or video)
    'FF++': (1000, FFTLDataset),
    'DFD': (363, DFDVideoDataset),
    'VSTL': (30, VSTLDataset),
    'DFTL': (133, DFTLDataset),
    'Davis2016-TL': (50, InpaintingDataset),
}


def train(genesis, dataloader_, test_loader_):
    # init
    device = genesis.device

    # running
    test_itr = enumerate(test_loader_)
    for epoch in range(PVT2Config.TRAIN_STEP):
        train_cache = TrainCache(size=16)
        _thread.start_new_thread(load_cache, (dataloader_, train_cache,))
        while not train_cache.finished:
            if train_cache.has_item():
                idx, item = train_cache.next_data()
                train_step(genesis, item, idx, epoch, device)
                time.sleep(6)
                test_step(genesis, idx, epoch, test_itr, device)
                time.sleep(2)


def train_step(genesis: Genesis, item: TrainItem, idx, epoch, device):
    src = cb2b(item.src, device)
    fake = cb2b(item.fake, device)
    masks = cb2b(item.masks, device)
    g = genesis.g([src, fake]).squeeze()
    g_loss = mask_loss(g, masks.squeeze())
    genesis.reset_grad()
    g_loss.backward()
    genesis.opt_g.step()
    # genesis.scheduler_g.step()
    if idx % 100 == 0:
        figureUtil.merge_pic(g, masks, 'images/{}_{}_mask.jpg'.format(data_type, idx))
        logger.info("Train Epoch:{}/{},G Loss:{:.5f}".format(epoch, idx, g_loss))


def test_step(genesis: Genesis, idx, epoch, test_itr, device):
    if idx % 100 == 0:
        genesis.eval()
        _, (label, _, sources, fakes, masks) = test_itr.__next__()
        # HashNet
        fakes = cb2b(fakes, device)
        sources = cb2b(sources, device)
        masks = cb2b(masks, device)
        g = genesis.g([sources, fakes]).squeeze()
        g_loss = fn.mask_loss(g, masks)
        logger.info("Test Epoch:{}/{},G Loss:{:.5f}".format(epoch, idx, g_loss))
        # save generate mask
        figureUtil.merge_pic(g, masks, '{}/{}_{}_test.jpg'.format(epoch, idx, data_type))
        # save generate mask
        genesis.save('model/{}_{}_'.format(epoch, idx))
        genesis.train()


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'/home/adminis/ppf/dataset/inpainting')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--type', type=str, default='Davis2016-TL')
parser.add_argument('--pretrained', type=str, default='/home/adminis/ppf/vrfx/models/davis')
parser.add_argument('--train_h', type=bool, default=False)
if __name__ == '__main__':
    args = parser.parse_args()
    data_type = args.type
    print('args:{}'.format(args))
    PVT2Config.NUM_CLASSES, Dataset = choices[args.type]
    genesis = Genesis(args.pretrained, args.local_rank, [args.local_rank], data_type=args.type, train_h=False)

    dataloader = get_dataloader(set_path=args.path, Dataset=Dataset, train_h=args.train_h)
    test_loader = get_dataloader(mode=PVT2Config.TEST, set_path=args.path, Dataset=Dataset, train_h=args.train_h)
    train(genesis, dataloader, test_loader)
