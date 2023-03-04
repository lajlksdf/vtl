import os
import sys

import cv2


class CountFile:
    file_nums = 0

    def count(self, dir_):
        if os.path.isdir(dir_):
            for d in os.listdir(dir_):
                self.count(os.path.join(dir_, d))
        elif dir_.endswith('mp4'):
            self.file_nums += 1


class CountFrame:
    frame_nums = 0
    sizes = []

    def video_info(self, dir_):
        if os.path.isdir(dir_):
            for d in os.listdir(dir_):
                self.video_info(os.path.join(dir_, d))
        elif dir_.endswith('mp4'):
            cap = cv2.VideoCapture(dir_)
            self.frame_nums += cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.sizes.append((cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


if __name__ == '__main__':
    path = sys.argv[1]
    path = os.path.abspath(path)
    cf = CountFile()
    cf.count(dir_=path)
    print(cf.file_nums)
    # cf2 = CountFrame()
    # cf2.video_info(dir_=path)
    # print(cf2.frame_nums)
    # print(cf2.sizes)
