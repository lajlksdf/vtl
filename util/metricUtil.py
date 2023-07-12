import os

from PIL import Image, ImageFilter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import json


def acc(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print("准确率：", accuracy)
    return accuracy


def precision_(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    print("精确率：", precision)

    return precision


def f1(y_true, y_pred):
    f1_ = f1_score(y_true, y_pred)
    print("F1值：", f1)
    return f1_


def recall_(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    print("召回率：", recall)
    return recall


def auc_(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    print("AUC值：", auc)
    return auc


class RobustnessTest:
    DETAIL = 0
    GAUSSIAN = 1
    BLUR = 2
    MedianFilter = 3
    CROP = 4
    NONE = -1


def robust_op(file, op=RobustnessTest.NONE):
    _f = os.path.join(file)
    im = Image.open(_f)
    w, h = im.size
    if op == 4:
        x, y = int(w * 0.05), int(h * 0.05)
        x_, y_ = int(w * 0.95), int(h * 0.95)
        im = im.crop((x, y, x_, y_))
    if op == 0:
        im = im.filter(ImageFilter.DETAIL)
    elif op == 1:
        im = im.filter(ImageFilter.GaussianBlur)
    elif op == 2:
        im = im.filter(ImageFilter.BLUR)
    elif op == 3:
        im = im.filter(ImageFilter.MedianFilter)
    return im


def write_json(y_pred, file_path):
    with open(file_path, 'w') as file:
        json.dump(y_pred, file)


def read_json(file_path):
    with open(file_path, 'r') as file:
        loaded_array = json.load(file)
        return loaded_array


if __name__ == '__main__':
    for i in range(-1, 5):
        print(i)
