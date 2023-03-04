import os

from torch.utils import data as tud

from config import PVT2Config
from dataset.BaseDataset import DataItem, BaseDataset


class FFTLDataset(BaseDataset):
    train_compresses = ['raw', 'c23', 'c40']
    test_compresses = ['raw']
    trace_listdir = ['faceswap', 'face2face', 'deepfakes', 'neuraltextures']
    test_listdir = ['faceshifter']

    def __init__(self, **kwargs):
        self.train_h = kwargs.get('train_h', False)
        if not self.train_h:
            FFTLDataset.train_compresses = ['raw', 'c40']
            FFTLDataset.test_compresses = ['c23']
            FFTLDataset.trace_listdir = ['faceswap']  # , 'face2face', 'deepfakes', 'neuraltextures'
            FFTLDataset.test_listdir = FFTLDataset.trace_listdir
        super().__init__(**kwargs)

    def _load_data(self):
        start = 0
        listdir = FFTLDataset.trace_listdir if self.mode == PVT2Config.TRAIN else FFTLDataset.test_listdir
        compresses = FFTLDataset.train_compresses if self.mode == PVT2Config.TRAIN else FFTLDataset.test_compresses
        item_path = self.set_path
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            for item in os.listdir(src_dir):
                src = os.path.join(src_dir, item)
                label = item
                fakes, masks = [], []
                for cls in listdir:
                    for c in compresses:
                        fake = os.path.join(fake_dir, cls, c, item)
                        fakes.append(fake)
                        if not self.train_h:
                            mask = os.path.join(mask_dir, cls, item)
                            masks.append(mask)
                data_item = DataItem(src, fakes, masks, label, start)
                start = data_item.end
                self.data.append(data_item)
        self.length = start // PVT2Config.FRAMES_STEP


def get_ff_dataloader(set_path, mode=PVT2Config.TRAIN,
                      num_workers=4,
                      batch_size=PVT2Config.BATCH_SIZE, test_op=-1, shuffle=True):
    dataset = FFTLDataset(set_path=set_path, mode=mode, test_op=test_op, type=1)
    dataloader = tud.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader
