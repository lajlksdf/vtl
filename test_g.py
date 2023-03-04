import argparse

import torch
from einops import rearrange

from config import PVT2Config
from dataset.BaseDataset import get_dataloader
from dataset.DFD_Video import DFDVideoDataset
from dataset.DFTL import DFTLDataset
from dataset.inpainting_dataset import InpaintingDataset
from dataset.splicingtl import VSTLDataset
from layer import helper
from layer.helper import read_imgs, to_list, read_masks
from layer.localizator import Localizator
from util import figureUtil
from util.logUtil import logger

choices = {
    # (number of original videos, Dataset: read frames or video)
    'DFD': (363, DFDVideoDataset, '/src/c23/videos/'),
    'VSTL': (30, VSTLDataset, '/train/src/'),
    'DFTL': (133, DFTLDataset, '/train/'),
    'Davis2016-TL': (50, InpaintingDataset, '/train/src/'),
}


def test():
    iou_total, total = 0, 0
    logger.info(f'Starting test: {type_}')
    dataloader = get_dataloader(mode=PVT2Config.TEST, set_path=args.path, Dataset=Dataset, shuffle=False, type='files',
                                train_h=args.train_h)
    total = len(dataloader)
    for idx, (label, src_files, fake_files, mask_files, _) in enumerate(dataloader):
        src = read_imgs(to_list(src_files)).to(device)
        src = rearrange(src, '(b t) ... -> b t ...', t=PVT2Config.NUM_FRAMES)
        fake = read_imgs(to_list(fake_files)).to(device)
        fake = rearrange(fake, '(b t) ... -> b t ...', t=PVT2Config.NUM_FRAMES)
        masks = read_masks(to_list(mask_files)).to(device)
        masks = rearrange(masks, '(b t) ... -> b t ...', t=PVT2Config.NUM_FRAMES)
        outs = net_g([src, fake]).squeeze()
        masks = masks.unsqueeze(dim=2).cpu().clone().detach()
        outs = outs.unsqueeze(dim=2).cpu().clone().detach()
        iou = helper.cal_iou(masks, outs)
        iou_total += iou
        step = int(100 * idx / total)
        str1 = '\r[%3d%%] %s' % (step, '>' * step)
        print(f'{str1}', end='', flush=True)
        figureUtil.merge_images(src_files, fake_files, mask_files, outs, '{}/{}_mask.jpg'.format(type_, idx))
    logger.info(f"Test-{type_}:{model}, iou:{iou_total / total:.5f}")


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'/home/adminis/ppf/dataset/DeepFakeDetection')
parser.add_argument('--pretrained', type=str, default='/home/adminis/ppf/vrfx/models/dfd/net_g.pth')
parser.add_argument('--type', type=str, default='DFD')
parser.add_argument('--local_rank', type=str, default='0')
parser.add_argument('--train_h', type=bool, default=False)
if __name__ == '__main__':
    args = parser.parse_args()
    type_ = args.type
    path_ = args.path
    model = args.pretrained
    device = torch.device(f"cuda:{args.local_rank}")
    PVT2Config.NUM_CLASSES, Dataset, src_path = choices[args.type]
    net_g = Localizator()
    print(f'loading {model}-{path_}')
    net_g.load_state_dict(torch.load(model, map_location=device))
    net_g = net_g.to(device)
    net_g.eval()
    with torch.no_grad():
        PVT2Config.FRAMES_STEP = PVT2Config.NUM_FRAMES * 100
        test()
