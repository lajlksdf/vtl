import os

from torch.utils import data as tud

from config import PVT2Config
from dataset.BaseDataset import BaseDataset, DataItem
from layer.helper import load_label_classes


class VSTLDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_labels(self):
        if self.mode == PVT2Config.TRAIN:
            load_label_classes(os.path.join(self.set_path, PVT2Config.TRAIN, 'src'))
        self.set_path = os.path.join(self.set_path, self.mode)

    def _load_data(self):
        start = 0
        item_path = os.path.abspath(self.set_path)
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            if self.mode == PVT2Config.TRAIN:
                listdir = sorted(os.listdir(fake_dir))
                for cls in listdir:
                    mask_ = os.path.join(mask_dir, cls)
                    fake_ = os.path.join(fake_dir, cls)
                    src = os.path.join(src_dir, cls)
                    fakes, masks = [], []
                    for _f in os.listdir(fake_):
                        mask = os.path.join(mask_, _f)
                        masks.append(mask)
                        fake = os.path.join(fake_, _f)
                        fakes.append(fake)
                    data_item = DataItem(src, fakes, masks, cls, start)
                    start = data_item.end
                    self.data.append(data_item)
            else:
                listdir = sorted(os.listdir(fake_dir))
                for _f in listdir:
                    label = _f
                    mask = os.path.join(mask_dir, _f)
                    fake = os.path.join(fake_dir, _f)
                    src = os.path.join(src_dir, _f)
                    data_item = DataItem(src=src, label=label, start=start, mask=[mask], fake=[fake])
                    start = data_item.end
                    self.data.append(data_item)
        self.length = start // PVT2Config.FRAMES_STEP


def get_vstl_dataloader(set_path, mode=PVT2Config.TRAIN, num_workers=min(os.cpu_count(), PVT2Config.BATCH_SIZE),
                        batch_size=PVT2Config.BATCH_SIZE, test_op=-1, shuffle=True):
    dataset = VSTLDataset(set_path=set_path, mode=mode, test_op=test_op)
    dataloader = tud.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader
