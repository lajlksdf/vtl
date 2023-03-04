import os

from torch.utils import data as tud

from config import PVT2Config
from dataset.BaseVideoDataset import BaseVideoDataset, VideoDataItem
from layer.helper import load_label_classes

train_compresses = ['c23']
test_compresses = ['c40']


class DFDVideoDataset(BaseVideoDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_labels(self):
        if self.mode == PVT2Config.TRAIN:
            load_label_classes(os.path.join(self.set_path, 'src', 'c23', 'videos'))

    def _load_data(self):
        item_path = self.set_path
        compresses = train_compresses if self.mode == PVT2Config.TRAIN else test_compresses
        start = 0
        src_dir = os.path.join(item_path, 'src', 'c23', 'videos')
        fake_dir = os.path.join(item_path, 'fake')
        mask_dir = os.path.join(item_path, 'masks', 'videos')
        for item in sorted(os.listdir(src_dir)):
            src = os.path.join(src_dir, item)
            label = item.replace('.mp4', '')
            fakes, masks = [], []
            for c in compresses:
                fake_video_dir = os.path.join(fake_dir, c, 'videos', label)
                for f in os.listdir(fake_video_dir):
                    fake_video = os.path.join(fake_video_dir, f)
                    fakes.append(fake_video)
                    mask_video_dir = os.path.join(mask_dir, f)
                    masks.append(mask_video_dir)
            if len(fakes) != 0:
                data_item = VideoDataItem(src, fakes, masks, label, start)
                start = data_item.end
                self.data.append(data_item)
        self.length = start // PVT2Config.FRAMES_STEP


def get_DFD_dataloader(set_path, mode=PVT2Config.TRAIN,
                       num_workers=4,
                       batch_size=PVT2Config.BATCH_SIZE, test_op=-1, shuffle=True):
    dataset = DFDVideoDataset(set_path=set_path, mode=mode, test_op=test_op, type=1)
    dataloader = tud.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader
