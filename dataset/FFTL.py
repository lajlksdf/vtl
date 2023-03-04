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
