import os

from torch.utils import data as tud

from config import PVT2Config
from dataset.BaseVideoDataset import BaseVideoDataset, VideoDataItem
from layer.helper import load_label_classes


class CelebDFVideoDataset(BaseVideoDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_labels(self):
        if self.mode == PVT2Config.TRAIN:
            load_label_classes(os.path.join(self.set_path, 'src'))

    def _load_data(self):
        item_path = self.set_path
        start = 0
        src_dir = os.path.join(item_path, 'src')
        for item in sorted(os.listdir(src_dir)):
            label = item.replace('.mp4', '')
            fake_dir = os.path.join(item_path, 'fake', label)
            src = os.path.join(src_dir, item)
            fakes, masks = [], []
            fake_videos = sorted(os.listdir(fake_dir))
            if len(fake_videos) > 2:
                list_v = fake_videos[:-2] if self.mode == PVT2Config.TRAIN else fake_videos[-2:]
            else:
                list_v = fake_videos
            for v in list_v:
                fake_video = os.path.join(fake_dir, v)
                fakes.append(fake_video)
            if len(fakes) != 0:
                data_item = VideoDataItem(src, fakes, masks, label, start)
                start = data_item.end
                self.data.append(data_item)
        self.length = start // PVT2Config.FRAMES_STEP


def get_CelebDF_dataloader(set_path, mode=PVT2Config.TRAIN,
                           num_workers=min(os.cpu_count(), PVT2Config.BATCH_SIZE),
                           batch_size=PVT2Config.BATCH_SIZE, test_op=-1, shuffle=True):
    dataset = CelebDFVideoDataset(set_path=set_path, mode=mode, test_op=test_op, type=1)
    dataloader = tud.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader
