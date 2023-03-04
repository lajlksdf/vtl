import os

from torch.utils import data as tud

from config import PVT2Config
from dataset.BaseDataset import BaseDataset, DataItem
from layer.helper import load_label_classes


class InpaintingDataset(BaseDataset):
    def __init__(self, **kwargs):
        super(InpaintingDataset, self).__init__(**kwargs)

    def _load_labels(self):
        if self.mode == PVT2Config.TRAIN and self.train_h:
            load_label_classes(os.path.join(self.set_path, PVT2Config.TRAIN, 'src'))
        self.set_path = os.path.join(self.set_path, self.mode)

    def _load_data(self):
        start = 0
        item_path = os.path.abspath(self.set_path)
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            listdir = sorted(os.listdir(src_dir))
            fake_list = sorted(os.listdir(fake_dir))
            for _f in listdir:
                label = _f
                mask = os.path.join(mask_dir, _f)
                fakes, masks = [], []
                src = os.path.join(src_dir, _f)
                for fake_ in fake_list:
                    fake = os.path.join(fake_dir, fake_, _f)
                    fakes.append(fake)
                    masks.append(mask)
                data_item = DataItem(src, fakes, masks, label, start)
                start = data_item.end
                self.data.append(data_item)
        self.length = start // PVT2Config.FRAMES_STEP


def get_inpainting_dataloader(set_path, mode=PVT2Config.TRAIN,
                              num_workers=min(os.cpu_count(), PVT2Config.BATCH_SIZE),
                              batch_size=PVT2Config.BATCH_SIZE, test_op=-1, shuffle=True):
    dataset = InpaintingDataset(set_path=set_path, mode=mode, test_op=test_op, type=1)
    dataloader = tud.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader
