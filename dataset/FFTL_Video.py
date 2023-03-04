import os
import random

from torch.utils import data as tud

from config import PVT2Config
from dataset.BaseVideoDataset import BaseVideoDataset, VideoDataItem

# compresses = ['c40']

compresses = ['raw', 'c23', 'c40']
trace_listdir = ['Face2Face', 'FaceSwap', 'Deepfakes', 'NeuralTextures']
test_listdir = ['FaceShifter']


# test_listdir = ['faceshifter', 'face2face', 'faceswap', 'deepfakes']
# trace_listdir = ['neuraltextures']


class FFVideoDataset(BaseVideoDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        start, video_data = self._get_video(index)
        idx = random.randint(0, 100) % len(video_data.fake_dir)
        i = random.randint(-3, 100)

        mask_data = self.read_video(video_data.mask_dir[idx], start, mask=True, op=i)
        fake_data = self.read_video(video_data.fake_dir[idx], start, op=i)
        return video_data.label, fake_data, mask_data, []

    def _load_data(self):
        listdir = trace_listdir if self.mode == PVT2Config.TRAIN else test_listdir
        item_path = self.set_path
        start = 0
        src_dir = os.path.join(item_path, 'original_sequences')
        fake_dir = os.path.join(item_path, 'manipulated_sequences')
        for item in sorted(os.listdir(src_dir)):
            src = os.path.join(src_dir, item)
            label = item
            fakes, masks = [], []
            for cls in listdir:
                mask_video = os.path.join(fake_dir, cls, 'masks', 'videos', item)
                for c in compresses:
                    fake_video = os.path.join(fake_dir, cls, c, 'videos', item)
                    fakes.append(fake_video)
                    masks.append(mask_video)
            if len(fakes) != 0:
                data_item = VideoDataItem(src, fakes, masks, label, start)
                start = data_item.end
                self.data.append(data_item)
        self.length = start // PVT2Config.FRAMES_STEP


def get_ff_dataloader(set_path, mode=PVT2Config.TRAIN,
                      num_workers=min(os.cpu_count(), PVT2Config.BATCH_SIZE),
                      batch_size=PVT2Config.BATCH_SIZE, test_op=-1, shuffle=True):
    dataset = FFVideoDataset(set_path=set_path, mode=mode, test_op=test_op, type=1)
    dataloader = tud.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader
