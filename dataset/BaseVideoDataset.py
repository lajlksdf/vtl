import os.path
import random
import uuid
from abc import ABCMeta

import numpy as np
import torch
from PIL import Image
from cv2 import cv2

from config import GlobalConfig, PVT2Config
from dataset.BaseDataset import BaseDataset
from layer.helper import to_mask_tensor_cv2, torch_resize


class VideoDataItem(object):
    def __init__(self, src, fake, mask, label, start):
        self.label = label
        self.start = start
        self.src_dir = src
        self.mask_dir = mask
        self.fake_dir = fake
        self.init_videos()
        # logger.info(f'{self.min_frame}-{label}-{len(fake)}-{fake[-1]}')

    def init_videos(self):
        min_frame = self.count_frame(-1, self.src_dir)
        for fake_dir in self.fake_dir:
            min_frame = self.count_frame(min_frame, fake_dir)
        for mask_dir in self.mask_dir:
            min_frame = self.count_frame(min_frame, mask_dir)
        self.min_frame = min_frame
        self.end = self.start + min_frame - 1

    def count_frame(self, min_frame, video):
        if not os.path.exists(video):
            return min_frame
        mask_cap = cv2.VideoCapture(video)
        frame_count = int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mask_cap.release()
        if min_frame == -1:
            return frame_count
        return min(min_frame, frame_count)


class BaseVideoDataset(BaseDataset, metaclass=ABCMeta):
    def __len__(self):
        return self.length

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_data(self):
        pass

    def __getitem__(self, index):
        while True:
            try:
                index = random.randint(0, self.length)
                return self.getitem(index)
            except:
                continue

    def getitem(self, index):
        start, video_data = self._get_video(index)
        idx = random.randint(0, 100) % len(video_data.fake_dir)
        i = random.randint(-3, 100)
        src, src_imgs = self.read_video(video_data.src_dir, start, op=i)
        fake, fake_files = self.read_video(video_data.fake_dir[idx], start, op=i)
        mask, mask_files = self.read_video(video_data.mask_dir[idx], start, mask=True, op=i)
        if self.mode == PVT2Config.TRAIN and self.train_h:
            fake_, _ = self.read_video(video_data.fake_dir[idx], start, op=i + 1)
            hashes = torch.cat([src, fake, fake_], dim=0)
            mask = ''
            return video_data.label, hashes, src, fake, mask
        elif self.type == 'files':
            return video_data.label, src_imgs, fake_files, mask_files, ''
        else:
            return video_data.label, fake_files, src, fake, mask

    def _get_video(self, idx):
        video_data, start, end, size = None, 0, 0, PVT2Config.NUM_FRAMES
        for e in self.data:
            video_data: VideoDataItem = e
            start = idx * PVT2Config.FRAMES_STEP
            if start < video_data.end:
                if end > video_data.end:
                    start = video_data.end - size
                break
        start = start - video_data.start
        return start, video_data

    def read_video(self, video, start, op=0, mask=False):
        if not os.path.exists(video):
            print(f'NOT exists: {video}')
        if self.mode == GlobalConfig.TEST:
            op = 0
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        tensors, imgs, count, c_ratio, flip = [], [], 0, [cv2.IMWRITE_JPEG_QUALITY,
                                                          random.randint(60, 100)], random.randint(-1, 1)
        while cap.isOpened() and count < PVT2Config.NUM_FRAMES:
            ret, frame = cap.read()
            if self.mode == GlobalConfig.TEST:
                img = frame.copy()
                outfile = 'tmp/' + str(uuid.uuid1()) + '_tmp.jpg'
                cv2.imwrite(outfile, img)
                imgs.append(outfile)
            count += 1
            if mask:
                tensor = to_mask_tensor_cv2(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                if op % 3 == 1:
                    msg = cv2.imencode(".jpg", frame, c_ratio)[1]
                    msg = (np.array(msg)).tobytes()
                    frame = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
                if op % 6 == 5:
                    frame = cv2.detailEnhance(frame, sigma_s=80, sigma_r=0.3)
                elif op % 6 == 1:
                    frame = cv2.GaussianBlur(frame, (5, 5), 0)
                elif op % 6 == 2:
                    frame = cv2.flip(frame, flip)
                # elif op % 6 == 3:
                #     frame = cv2.medianBlur(frame, 5)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = torch_resize(image)
            tensors.append(tensor)
        cap.release()
        data = torch.cat(tensors, dim=0).unsqueeze(dim=0)
        return data, imgs
