import os
import random
import time

import torch
from PIL import Image as Image, ImageFilter
from torch.utils import data as tud
from torch.utils.data import Dataset

from config import PVT2Config
from layer.helper import load_label_classes, torch_resize, to_mask_tensor
from util.logUtil import logger


class DataItem(object):
    def __init__(self, src, fake, mask, label, start):
        self.label = label
        self.start = start
        self.src_dir = src
        self.mask_dir = mask
        self.fake_dir = fake
        self.init_files()
        print(f'{label}-{fake}-{mask}')

    def init_files(self):
        self.files = sorted(os.listdir(self.src_dir))
        if isinstance(self.fake_dir, list):
            for fake_dir in self.fake_dir:
                self.set_files(fake_dir)
            for mask_dir in self.mask_dir:
                self.set_files(mask_dir)
        else:
            self.set_files(self.fake_dir)
            self.set_files(self.mask_dir)
        self.end = self.start + len(self.files)

    def set_files(self, dir_):
        files = sorted(os.listdir(dir_))
        if len(files) < len(self.files):
            self.files = files


class BaseDataset(Dataset):
    def __len__(self):
        return self.length

    def __init__(self, **kwargs):
        super(BaseDataset, self).__init__()
        # 数据指针，k:label,v:start index
        self.set_path = os.path.abspath(kwargs.get('set_path', PVT2Config.SET_PATH))
        self.batch_size = kwargs.get('batch_size', PVT2Config.BATCH_SIZE)
        self.mode = kwargs.get('mode', PVT2Config.TRAIN)
        self.test_op = kwargs.get('test_op', -1)
        self.type = kwargs.get('type', '')
        self.train_h = kwargs.get('train_h', False)
        self._load_labels()

        self.data = []
        self._load_data()
        logger.info(f'Dataset,loading.{self.length}-{self.mode}-{self.batch_size}-{self.set_path}-{self.test_op}')

    def count(self, count):
        self.length = count // self.batch_size // PVT2Config.FRAMES_STEP

    def _load_labels(self):
        if self.mode == PVT2Config.TRAIN:
            load_label_classes(os.path.join(self.set_path, PVT2Config.TRAIN))
        self.set_path = os.path.join(self.set_path, self.mode)

    def _load_data(self):
        pass

    def _get_files(self, idx):
        video_data, start, end, size = None, 0, 0, PVT2Config.NUM_FRAMES
        for e in self.data:
            video_data: DataItem = e
            # global length
            start = idx * PVT2Config.FRAMES_STEP
            if start < video_data.end:
                end = start + size
                if end > video_data.end:
                    # item length
                    start = video_data.end - size
                    end = video_data.end
                break
        start = start - video_data.start
        end = end - video_data.start
        files = video_data.files[start:end]
        return files, video_data

    def __getitem__(self, index):
        files, video_data = self._get_files(index)
        idx = random.randint(0, 100) % len(video_data.fake_dir)
        if self.train_h:
            mask_data = ''
        else:
            mask_data = read_data(video_data.mask_dir[idx], files, op=(4 if self.test_op == 4 else -1), mask=True)
        if self.mode == PVT2Config.TRAIN and self.train_h:
            i = random.randint(-3, 5)
            src = read_data(video_data.src_dir, files, op=i)

            fake_data = read_data(video_data.fake_dir[idx], files, op=i)
            idx = random.randint(0, 100) % len(video_data.fake_dir)
            fake_data2 = read_data(video_data.fake_dir[idx], files, op=i)
            hashes = [src, fake_data, fake_data2]
            return video_data.label, torch.cat(hashes), src, fake_data, mask_data
        elif self.type == 'files':
            src_files, fake_files, mask_files = [], [], []
            for f in files:
                src_files.append(os.path.join(video_data.src_dir, f))
                fake_files.append(os.path.join(video_data.fake_dir[idx], f))
                mask_files.append(os.path.join(video_data.mask_dir[idx], f))
            return video_data.label, src_files, fake_files, mask_files, ''
        else:
            fake_files = []
            for f in files:
                fake_files.append(os.path.join(video_data.fake_dir[idx], f))
            src = read_data(video_data.src_dir, files)
            fake = read_data(video_data.fake_dir[idx], files, op=self.test_op)
            return video_data.label, fake_files, src, fake, mask_data


def read_data(_dir, files, op=-1, mask=False):
    try:
        tensors = []
        for f in files:
            _f = os.path.join(_dir, f)
            im = Image.open(_f)
            w, h = im.size
            if op == 4:
                x, y = int(w * 0.05), int(h * 0.05)
                x_, y_ = int(w * 0.95), int(h * 0.95)
                im = im.crop((x, y, x_, y_))
            if op == 0:
                im = im.filter(ImageFilter.DETAIL)
            elif op == 1:
                im = im.filter(ImageFilter.GaussianBlur)
            elif op == 2:
                im = im.filter(ImageFilter.BLUR)
            elif op == 3:
                im = im.filter(ImageFilter.MedianFilter)
            if mask:
                tensor = to_mask_tensor(im, PVT2Config.IMAGE_SIZE).unsqueeze(0)
            else:
                tensor = torch_resize(im, PVT2Config.IMAGE_SIZE)
            tensors.append(tensor)
        tensors = torch.cat(tensors, dim=0)
        return tensors.unsqueeze(0)
    except BaseException as e:
        logger.info('data read error:{},{},{}'.format(_dir, files, e))


class TrainItem:
    def __init__(self, label, hashes, src, fake, masks):
        super(TrainItem, self).__init__()
        self.label = label
        self.hashes = hashes
        self.src = src
        self.fake = fake
        self.masks = masks


class TrainCache:
    def __init__(self, size):
        self.cache = {}
        self.size = size
        self.finished = False

    def put(self, idx, item: TrainItem):
        self.cache[idx] = item

    def is_stop(self):
        return len(self.cache) > self.size

    def next_data(self):
        return self.cache.popitem()

    def has_item(self):
        return len(self.cache) > 0

    def finish(self):
        self.finished = True


def load_cache(dataloader, train_cache: TrainCache):
    # train cache
    for idx, (label, hashes, src, fake, masks) in enumerate(dataloader):
        cache = TrainItem(label, hashes, src, fake, masks)
        train_cache.put(idx, cache)
        while train_cache.is_stop():
            time.sleep(1)
    time.sleep(10)
    train_cache.finish()


def get_dataloader(set_path, Dataset, mode=PVT2Config.TRAIN, train_h=True,
                   batch_size=PVT2Config.BATCH_SIZE, test_op=-1, shuffle=True, type=''):
    num_workers = 4 if mode == PVT2Config.TRAIN else 0
    dataset = Dataset(set_path=set_path, mode=mode, test_op=test_op, train_h=train_h, type=type)
    dataloader = tud.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader
