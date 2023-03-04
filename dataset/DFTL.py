import os

from config import PVT2Config
from dataset.BaseDataset import BaseDataset, DataItem


class DFTLDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_data(self):
        start = 0
        listdir = os.listdir(self.set_path)
        for item in listdir:
            item_path = os.path.join(self.set_path, item)
            if os.path.isdir(item_path):
                src_dir = os.path.join(item_path, 'src')
                label = os.path.basename(item_path)
                fake_dir = os.path.join(item_path, 'fake')
                mask_dir = os.path.join(item_path, 'mask')
                listdir = sorted(os.listdir(fake_dir))
                fakes, masks = [], []
                for _f in listdir:
                    fake = os.path.join(fake_dir, _f)
                    mask = os.path.join(mask_dir, _f)
                    fakes.append(fake)
                    masks.append(mask)
                data_item = DataItem(src_dir, fakes, masks, label, start)
                start = data_item.end
                self.data.append(data_item)
        self.length = start // PVT2Config.FRAMES_STEP
