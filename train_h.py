import _thread
import argparse

import torch.nn as nn

from config import PVT2Config
from dataset.BaseDataset import TrainCache, load_cache, get_dataloader, TrainItem
from dataset.CelebDF_Video import CelebDFVideoDataset
from dataset.DFD_Video import DFDVideoDataset
from dataset.DFTL import DFTLDataset
from dataset.FFTL import FFTLDataset
from dataset.inpainting_dataset import InpaintingDataset
from dataset.splicingtl import VSTLDataset
from layer import fn
from layer import helper
from layer.genesis import Genesis
from layer.helper import cb2b, get_tensor_target
from util.logUtil import logger

# program init
bce_loss = nn.BCELoss(reduction='mean')
itr_times, g_losses, h_losses, d_losses, h_d_losses, accuracies, hash_dists = [], [], [], [], [], [], []
choices = {  # (number of original videos, Dataset: read frames or video)
    'FF++': (1000, FFTLDataset),
    'DFD': (363, DFDVideoDataset),
    'Celeb-DF': (590, CelebDFVideoDataset),
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
                test_step(genesis, idx, epoch, test_itr, device)


def train_step(genesis: Genesis, item: TrainItem, idx, epoch, device):
    hashes = cb2b(item.hashes, device)
    label = item.label
    d, h = genesis.h(hashes)
    h_loss = fn.hash_triplet_loss(h, label, d)
    # d loss
    d_label = get_tensor_target(label).to(device)
    d_loss = bce_loss(d.flatten(), d_label.flatten())
    d_h_loss = h_loss + d_loss
    # backward
    genesis.reset_grad()
    d_h_loss.backward()
    genesis.opt_h.step()
    # genesis.scheduler_h.step()
    # hashcode accuracy
    acc = helper.find_index(h, label)
    if idx % 100 == 0:
        logger.info("Train Epoch:{}/{},H Loss:{:.5f},D Loss:{:.9f},hash dis:{:.5f} acc:{:.5f}".
                    format(epoch, idx, h_loss, d_loss, helper.hash_intra_dis(), acc))


def test_step(genesis: Genesis, idx, epoch, test_itr, device):
    if idx % 100 == 0:
        genesis.eval()
        _, (label, _, sources, fakes, masks) = test_itr.__next__()
        # HashNet
        fakes = cb2b(fakes, device)
        h = genesis.h(fakes)
        acc = helper.find_index(h, label)
        # epoch log
        logger.info("Test :{}/{}, acc:{:.5f}".format(epoch, idx, acc))
        genesis.save('model/{}_{}_'.format(epoch, idx))
        helper.save_hash('model/{}_{}_'.format(epoch, idx), genesis.hash_bits)
        genesis.train()


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'/home/adminis/ppf/dataset/DFTL')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--type', type=str, default='DFTL')
parser.add_argument('--pretrained', type=str, default='')
parser.add_argument('--hash_bits', type=int, default=PVT2Config.HASH_BITS)
if __name__ == '__main__':
    args = parser.parse_args()
    print('args:{}'.format(args))
    PVT2Config.HASH_BITS = args.hash_bits
    PVT2Config.NUM_CLASSES, Dataset = choices[args.type]
    genesis = Genesis(args.pretrained, args.local_rank, [args.local_rank], data_type=args.type)

    dataloader = get_dataloader(set_path=args.path, Dataset=Dataset)
    test_loader = get_dataloader(mode=PVT2Config.TEST, set_path=args.path, Dataset=Dataset)
    train(genesis, dataloader, test_loader)
