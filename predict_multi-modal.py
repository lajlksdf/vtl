import argparse
import json
import os
import cv2

import torch
from PIL import Image
from torch import Tensor

from config import PVT2Config
from layer.helper import tensor_to_binary, compute_hamming_dist, file2tensor, torch_resize
from layer.vit_hash import PVT2HashNet
from util.metricUtil import robust_op, RobustnessTest, write_json

# device = torch.device("cpu")
hashmap = {}
result_map = {}


def load_map(file):
    try:
        if os.path.exists(file):
            with open(file, 'r') as f:
                print(f'loading:{file}')
                content = json.load(f)
                hashmap.update(content)
        return True
    except BaseException as e:
        print(e)
        return False


def load_model(model_path_, device_):
    PVT2Config.NUM_CLASSES = len(hashmap)
    print(PVT2Config.NUM_CLASSES)
    net_h = PVT2HashNet()
    net_h.load_state_dict(torch.load(model_path_, map_location=device_))
    net_h = net_h.to(device_)
    net_h.eval()
    return net_h


def find_index(hashset: Tensor, label_):
    hash_ = tensor_to_binary(hashset).cpu()
    v_ = hash_.detach()[0].numpy()
    min_dis = PVT2Config.HASH_BITS
    find_idx = 0
    for k, v in hashmap.items():
        dis = compute_hamming_dist(v, v_)
        if dis < min_dis:
            min_dis = dis
            find_idx = k
    res = str(find_idx) == str(label_)
    if not res:
        print(f'find_idx:{find_idx}-{label_}')
    return res


def predict(image_path_, label_, net_h, op):
    image = robust_op(image_path_, op)
    x = torch_resize(image)
    y = x.repeat(PVT2Config.NUM_FRAMES, 1, 1, 1).unsqueeze(0)
    h = net_h(y.to(device))
    return find_index(h, label_)


def predict_cv2(image, label_, net_h):
    img_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv2)
    x = torch_resize(img_pil)
    y = x.repeat(PVT2Config.NUM_FRAMES, 1, 1, 1).unsqueeze(0)
    h = net_h(y.to(device))
    return find_index(h, label_)


y_pred = []


# image_path like dir_path/label/x.jpg
def test_read_image(op):
    count, real = 0, 0
    for p in os.listdir(dir_path):
        label = p
        path_ = os.path.join(dir_path, p)
        print(f'predict:{path_}')
        images = os.listdir(path_)
        images_ = [images[0], images[len(images) // 2], images[-1]]
        for f in images_:
            count += 1
            image_path = os.path.join(path_, f)
            if predict(image_path, label, model, op):
                real += 1
                y_pred.append(1)
            else:
                y_pred.append(0)

    print(f'real:{real}-count:{count}, ACC:{real / count}')
    write_json(y_pred, f'{dir_path}-{op}.json')


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'')
parser.add_argument('--pretrained', type=str, default='')
parser.add_argument('--local_rank', type=str, default='0')
parser.add_argument('--hash_path', type=str, default='')
if __name__ == '__main__':
    '''Multi-modal source tracing, using image tracing video.'''
    args = parser.parse_args()
    dir_path = args.path
    model_path = args.pretrained
    hash_path = args.hash_path
    device = torch.device(f"cuda:{args.local_rank}")
    load_map(hash_path)
    model = load_model(model_path, device)
    for i in range(-1, 5):
        test_read_image(i)
