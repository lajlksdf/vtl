import json
import math
import os
import time
import uuid
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image as Image
from einops import rearrange
from torch import Tensor, nn as nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision.transforms import transforms

from config import PVT2Config

loader = transforms.Compose([
    transforms.ToTensor(),
])
unloader = transforms.ToPILImage()
hashmap = {}


def set_hash_bits(hb):
    PVT2Config.HASH_BITS = hb


def get_hash_bits():
    return PVT2Config.HASH_BITS


def get_net_h(hash_bit=get_hash_bits()):
    return str(hash_bit) + '_net_h.pth'


def get_hash_json(hash_bit=get_hash_bits()):
    return str(hash_bit) + '_hash.json'


def gen_hash(hashset: []):
    """
    hash list to hash by select most common element
    Args:
        hashset: must be numpy array

    Returns:

    """
    arr = np.array(hashset)
    _hash = []
    for i in range(get_hash_bits()):
        bit_ = Counter(arr[:, i]).most_common(1)
        _hash.append(int(bit_[0][0]))
    return _hash


def to_binary(code):
    hashset = []
    code = np.asarray(code)
    for h in code:
        hash_code = []
        mean_v = h.mean()
        for i in h:
            if i > mean_v:
                hash_code.append(1)
            else:
                hash_code.append(0)
        hashset.append(hash_code)
    return hashset


def update_hash(h_map: {}):
    hash_bits = get_hash_bits()
    for k, v in h_map.items():
        hashmap[k] = v
        hash_bits = len(v)
    save_hash(hash_bits=hash_bits)


def get_hash(label):
    load_hash()
    if not hashmap.__contains__(label):
        hashmap[label] = np.random.randint(0, 2, get_hash_bits()).tolist()
    return hashmap[label]


def to_image(image):
    return unloader(image)


def save_hash(prefix='./', hash_bits=get_hash_bits()):
    _save(hashmap, prefix + get_hash_json(hash_bits))


def _save(map_, file):
    while not __save(map_, file):
        time.sleep(2)


def __save(map_, file):
    try:
        with open(file, 'w') as f:
            json.dump(map_, f)
        return True
    except BaseException as e:
        print(e)
        return False


def _load(map_, file):
    while not __load(map_, file):
        time.sleep(2)


def __load(map_, file):
    try:
        if os.path.exists(file):
            with open(file, 'r') as f:
                content = json.load(f)
                map_.update(content)
        return True
    except BaseException as e:
        print(e)
        return False


def get_hashmap():
    load_hash()
    return hashmap


def load_hash(file=get_hash_json()):
    _load(hashmap, file)


def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hash_intra_dis():
    size = len(hashmap)
    intra_dis = 0
    for arr1 in hashmap.values():
        for arr2 in hashmap.values():
            if arr1 != arr2:
                intra_dis = intra_dis + compute_hamming_dist(arr1, arr2)
    return intra_dis / (size * (size - 1))


def hashmap_mean():
    np_arr = np.array([list(item) for item in hashmap.values()])
    return np_arr.mean()


def tensor_to_binary(x: Tensor, act='tahn'):
    """
    <br>1. torch.sign(x) # with relu
    <br>2. (torch.sign(x - 0.5)+ 1) / 2  # with sigmiod
    <br>3. (torch.sign(x) + 1) / 2  # with tahn
    :param x:
    :return:
    """
    if act == 'tahn':
        return (torch.sign(x) + 1) / 2
    elif act == 'sigmoid':
        return (torch.sign(x - 0.5) + 1) / 2  # with sigmiod
    elif act == 'relu':
        return torch.sign(x)


def compute_hamming_dist(a, b):
    """
    Computes hamming distance vector wisely
    Args:
        a: row-ordered codes {0,1}
        b: row-ordered codes {0,1}
    Returns:
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b))


def find_index(hashset: Tensor, label: []):
    """
    Save generate images all in one
    Args:
        label: data label
        hashset: generate hashcode
        name: save name

    Returns:

    """
    hashset_ = tensor_to_binary(hashset)
    h_map = to_hashmap(hashset_, label)
    real_count = 0
    for k_, v_ in h_map.items():
        f_l = k_
        min_dis = get_hash_bits()
        for k, v in hashmap.items():
            dis = compute_hamming_dist(v, v_)
            if dis < min_dis:
                min_dis = dis
                f_l = k
        if k_ == f_l:
            real_count += 1
    return real_count / len(label)


def to_hashmap(hashset: Tensor, label):
    hashset_ = rearrange(hashset.detach().clone(), '(c b) ... -> c b ...', c=len(label)).cpu()
    h_map = {}  # init
    for l_ in label:
        h_map[l_] = []
    for i in range(len(label)):
        l_ = label[i]
        h_map[l_].append(gen_hash(hashset_[i].numpy()))
    for k, v in h_map.items():
        h_map[k] = gen_hash(v)
    return h_map


def cb2b(x: Tensor, device=None):
    if device:
        return rearrange(x, 'c b ... -> (c b) ...').to(device)
    else:
        return rearrange(x, 'c b ... -> (c b) ...')


def compress(file, quality=85):
    '''
    compress image
    :param file: image file
    :param quality: 85 or 70
    :return: compressed image
    '''
    outfile = str(uuid.uuid1()) + '.jpg'
    im = Image.open(file)
    im.save(outfile, quality=quality)
    return outfile


def init_m(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


def to_mask_tensor_cv2(img, image_size=224):
    img = cv2.resize(img, (image_size, image_size))
    mask = torch.zeros([image_size, image_size])
    for h in range(image_size):
        for w in range(image_size):
            if img[h, w] > 50:
                mask[h, w] = 1.0
    return torch.unsqueeze(mask, 0)


def img2tensor(img):
    return loader(img)


def torch_resize(image, size=224):
    image = img2tensor(image).unsqueeze(0)
    image = F.interpolate(image, size=size, mode='bilinear', align_corners=True)
    return image


label_set = {}


def load_label_classes(data_path):
    classes = os.listdir(data_path)
    for c in classes:
        if os.path.isdir(os.path.join(data_path, c)):
            label_set[c] = 0
        elif c.__contains__('.mp4'):
            c = c.replace('.mp4', '')
            label_set[c] = 0
    num_classes = len(classes)
    print(label_set)
    return num_classes


def get_classes_label(label):
    l_set = label_set.copy()
    l_set[label] = 1
    return list(l_set.values())


def get_tensor_target(labels: []):
    ts = []
    for l in labels:
        ts.append(get_classes_label(l))
    x = np.asarray(ts, dtype=np.float32).repeat(3, axis=0)
    return torch.from_numpy(x)


def to_mask(img, th=0.3):
    img[img < th] = 0
    img[img > 0] = 1
    return img


def process_mask(file, bin=False):
    if bin:
        gray = file
    else:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_MASK)
    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=2)  # sure background area
    sure_op = cv2.erode(opening, kernel, iterations=2)  # sure foreground area
    sure_fg = cv2.erode(sure_bg, kernel, iterations=2)  # sure foreground area
    # cv2.imshow('1', to_binary(sure_fg))
    # cv2.imshow('2', to_binary(sure_op))
    # cv2.imshow('gray', gray)
    # cv2.waitKey(-1)
    return to_mask(sure_fg), to_mask(sure_op)


def get_binary_img(file, bin=False):
    if bin:
        img = file
    else:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bin_img = np.zeros(shape=(img.shape), dtype=np.uint8)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            bin_img[i][j] = 255 if img[i][j] > 85 else 0
    # cv2.imshow('bin_img', bin_img)
    bin_img = cv2.blur(bin_img, (5, 5))
    # cv2.waitKey(-1)
    return process_mask(bin_img, True)


def cal_iou(mask_data, outs: Tensor):
    iou = mask_iou(to_mask(mask_data), to_mask(outs))
    return iou


def mask_iou(mask1, mask2):
    area1 = mask1.sum()
    area2 = mask2.sum()
    inter = ((mask1 + mask2) == 2).sum()
    m_iou = inter / (area1 + area2 - inter)
    m_iou = round(m_iou.item(), 3)
    return 0 if math.isnan(m_iou) else m_iou


def tensor2img(image, path=None):
    image = unloader(image)
    if path:
        image.save(path)
    return image


def tensor_resize(image, size=224):
    image = F.interpolate(image, size=size, mode='bilinear', align_corners=True)
    return image


def read_imgs(files):
    tensors = []
    for f in files:
        tensor = file2tensor(f)
        tensors.append(tensor)
    tensors = torch.cat(tensors, dim=0)
    return tensors


def read_masks(files):
    tensors = []
    for f in files:
        im = Image.open(f)
        tensor = to_mask_tensor(im, PVT2Config.IMAGE_SIZE).unsqueeze(0)
        tensors.append(tensor)
    tensors = torch.cat(tensors, dim=0)
    return tensors


def file2tensor(file):
    im = Image.open(file)
    return torch_resize(im, PVT2Config.IMAGE_SIZE)


def to_list(files):
    ss = []
    b = len(files[0])
    for j in range(b):
        for i in range(PVT2Config.NUM_FRAMES):
            ss.append(files[i][j])
    return ss


def to_mask_tensor(img, image_size=PVT2Config.IMAGE_SIZE):
    img = img.convert('L')
    img = img.resize((image_size, image_size))
    img = np.asarray(img, dtype=np.int32)
    mask = torch.zeros([image_size, image_size])
    for h in range(image_size):
        for w in range(image_size):
            if img[h, w] > 50:
                mask[h, w] = 1.0
    return mask
