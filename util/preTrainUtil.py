import os

import cv2
import numpy as np

p = r'/ssd1/data/ppf/h'


class MeanStf:
    def __init__(self):
        self.TOTAL = 0
        self.means = [0, 0, 0]
        self.stdevs = [0, 0, 0]

    def mean(self, path):
        for line in os.listdir(path):
            a = os.path.join(path, line)
            if os.path.isdir(a) and not a.endswith('mask'):
                self.print_mean(a)
                self.mean(a)
            elif a.endswith('.jpg'):
                try:
                    self.TOTAL += 1
                    img = cv2.imread(a)
                    img = np.asarray(img)
                    img = img.astype(np.float32) / 255.
                    for i in range(3):
                        self.means[i] += img[:, :, i].mean()
                        self.stdevs[i] += img[:, :, i].std()
                except Exception as e:
                    print(a)
                    print(e)
                    continue

    def print_mean(self, path):
        if self.TOTAL > 0:
            print(path)
            means = self.means[:]
            means.reverse()
            stdevs = self.stdevs[:]
            stdevs.reverse()
            means = np.asarray(means) / self.TOTAL
            stdevs = np.asarray(stdevs) / self.TOTAL
            print("TOTAL={},normMean = {},normStd = {}".format(self.TOTAL, means, stdevs))


if __name__ == '__main__':
    x = MeanStf()
    x.mean(p)
    x.print_mean(p)
