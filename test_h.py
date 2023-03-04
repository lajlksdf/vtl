import argparse
import json
import os
from collections import Counter

import torch
from PIL import Image
from cv2 import cv2
from torch import Tensor

from config import PVT2Config
from dataset.BaseDataset import get_dataloader
from dataset.CelebDF_Video import CelebDFVideoDataset
from dataset.DFD_Video import DFDVideoDataset
from dataset.DFTL import DFTLDataset
from dataset.FFTL import FFTLDataset
from dataset.inpainting_dataset import InpaintingDataset
from dataset.splicingtl import VSTLDataset
from layer import helper
from layer.helper import tensor_to_binary, compute_hamming_dist, to_image
from layer.vit_hash import PVT2HashNet
from util.logUtil import logger

# device = torch.device("cpu")
hashmap = {}
result_map = {}
fake_imgs_map = {}

compresses = ['raw', 'c23', 'c40']
trace_listdir = ['face2face', 'faceswap', 'deepfakes', 'neuraltextures', 'faceshifter']
choices = {
    # (number of original videos, Dataset: read frames or video)
    'FF++': (1000, FFTLDataset, 'train/src/'),
    'DFD': (363, DFDVideoDataset, 'src/c23/videos/'),
    'Celeb-DF': (590, CelebDFVideoDataset, 'src/'),
    'VSTL': (30, VSTLDataset, 'train/src/'),
    'DFTL': (133, DFTLDataset, 'train/'),
    'Davis2016-TL': (50, InpaintingDataset, 'train/src/'),
}


def find_index(hashset: Tensor, label: []):
    find_labels = []
    hashset_ = tensor_to_binary(hashset).cpu()
    real_count = 0
    for i in range(len(label)):
        k_ = label[i]
        f_l = k_
        v_ = hashset_[i].numpy()
        min_dis = PVT2Config.HASH_BITS
        for k, v in hashmap.items():
            dis = compute_hamming_dist(v, v_)
            if dis < min_dis:
                min_dis = dis
                f_l = k
        if k_ == f_l:
            real_count += 1
        find_labels.append(f_l)
    return real_count / len(label), find_labels


def put_result(labels, find_labels, fake_images):
    for i in range(len(labels)):
        l = labels[i]
        l_ = find_labels[i]
        if result_map.__contains__(l):
            fake_imgs_map[l] = fake_images[i][0]
            result_map[l].append(l_)
        else:
            result_map[l] = [l_]


def test(datatype_, path, test_op, hash_path, h_model, cls):
    logger.info(f'Starting test: {cls}')
    dataloader = get_dataloader(mode=PVT2Config.TEST, set_path=args.path, Dataset=Dataset, shuffle=False,
                                train_h=args.train_h)
    total = len(dataloader)
    net_h = PVT2HashNet()
    load_map(hash_path)
    print(f'loading {h_model}-{hash_path}')
    net_h.load_state_dict(torch.load(h_model, map_location=device))
    net_h = net_h.to(device)
    net_h.eval()
    for idx, (label, fake_file, src, fakes, masks) in enumerate(dataloader):
        fakes = helper.cb2b(fakes, device)
        acc = test_h(fakes, label, net_h, fake_file)
        step = int(100 * idx / total)
        str1 = '[\r %3d %3d%% %s]' % (acc, step, '>' * step)
        print(f'[{str1}', end='', flush=True)
    cal_trace_acc(path, datatype_, test_op, cls)


def test_h(fakes, label, net_h, fake_images):
    h = net_h(fakes)
    acc, find_labels = find_index(h, label)
    put_result(label, find_labels, fake_images)
    return acc


def cal_trace_acc(path, datatype_, test_op, cls):
    acc = 0
    for k, v in result_map.items():
        data = Counter(v)
        k_ = data.most_common(1)[0][0]
        if k == k_:
            res = 'right'
            acc += 1
        else:
            res = 'err'
            logger.info(f'Error-{k_}-{k}')
        try:
            find_file = find_files(path, k_)
            fake_file = fake_imgs_map[k]
            merge_pic_trace([fake_file], [find_file], f'{datatype_}/{k}-{res}-trace.jpg')
        except:
            continue
    acc = acc / len(result_map)
    logger.info(f"Test-{cls}:{test_op}, hash_bits:{PVT2Config.HASH_BITS}, acc:{acc:.5f}")


def merge_pic_trace(find_files: [], files: [], name):
    h, w, b = 224, 224, len(find_files),
    images = Image.new('RGB', (w * b, h * 2))
    for i in range(b * 2):
        if i % 2 == 0:
            img = find_files[i // 2]
            if isinstance(img, str):
                img = Image.open(img)
        else:
            image = files[i // 2]
            img = Image.open(image)
        img = img.resize((224, 224))
        images.paste(img, box=((i // 2) * w, (i % 2) * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def find_files(path, l_):
    path_f = None
    if type_ == 'DFTL':
        path_f = os.path.join(path, src_path, l_, 'src')
    elif type_ == 'Davis2016-TL' or type_ == 'VSTL' or type_ == 'FF++':
        path_f = os.path.join(path, src_path, l_)
    if path_f:
        file = sorted(os.listdir(path_f))[0]
        file = os.path.join(path_f, file)
        return file
    else:
        return read_video_cover(path, l_)


def read_video_cover(path, l_):
    cover = f'tmp/{l_}_{type_}.jpg'
    if os.path.exists(cover):
        return cover
    path_f = os.path.join(path, src_path, l_) + '.mp4'
    print(path_f)
    vidcap = cv2.VideoCapture(path_f)
    success, image = vidcap.read()
    cv2.imwrite(cover, image)
    vidcap.release()
    return cover


def merge_pic(files: [], name):
    print(files)
    h, w, b = 224, 224, len(files)
    image = Image.open(files[0][0])
    images = Image.new(image.mode, (w * b, h))
    for i in range(b):
        image = files[i][0]
        img = Image.open(image)
        img = img.resize((224, 224))
        images.paste(img, box=(i * w, 0))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def merge_pic_test(g_tensor: Tensor, mask: Tensor, name):
    b, t, h, w = g_tensor.shape
    g_tensor = g_tensor.detach().cpu()
    img = to_image(g_tensor[0][0])
    images = Image.new(img.mode, (w * t, h * b * 2))
    for j in range(b * 2):
        for i in range(t):
            if j % 2 == 1:
                img = g_tensor[j // 2][i]
            else:
                img = mask[j // 2][i]
            img = to_image(img)
            images.paste(img, box=(i * w, j * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


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


def robust_test():
    with torch.no_grad():
        PVT2Config.FRAMES_STEP = PVT2Config.NUM_FRAMES * 10
        PVT2Config.HASH_BITS = hash_bits
        FFTLDataset.test_listdir = [test_cls]
        FFTLDataset.test_compresses = [test_compresses]
        json_path = os.path.join(h_path, f'{hash_bits}_hash.json')
        model_path = os.path.join(h_path, f'{hash_bits}_net_h.pth')
        for i in range(-1, 5):
            result_map.clear()
            test(type_, path_, i, json_path, model_path, 'robust')


def normal_test():
    with torch.no_grad():
        PVT2Config.FRAMES_STEP = PVT2Config.NUM_FRAMES * 5
        json_path = os.path.join(h_path, f'{hash_bits}_hash.json')
        model_path = os.path.join(h_path, f'{hash_bits}_net_h.pth')
        result_map.clear()
        test(type_, path_, -1, json_path, model_path, 'normal-' + type_)


def cross_test():
    with torch.no_grad():
        PVT2Config.FRAMES_STEP = PVT2Config.NUM_FRAMES * 10
        PVT2Config.HASH_BITS = hash_bits
        for model_cls in trace_listdir:
            json_path = os.path.join(h_path, model_cls, '512_hash.json')
            model_path = os.path.join(h_path, model_cls, '512_net_h.pth')
            t_listdir = trace_listdir[:]
            for t_cls in t_listdir:
                FFTLDataset.test_listdir = [t_cls]
                result_map.clear()
                test(type_, path_, -1, json_path, model_path, model_cls + '-' + t_cls)


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'/home/adminis/ppf/dataset/inpainting')
parser.add_argument('--pretrained', type=str, default='/home/adminis/ppf/vrfx/models/davis')
parser.add_argument('--model_cls', type=str, default='face2face')
parser.add_argument('--test_cls', type=str, default='faceshifter')
parser.add_argument('--c', type=str, default='raw')
parser.add_argument('--type', type=str, default='Davis2016-TL')
parser.add_argument('--local_rank', type=str, default='0')
parser.add_argument('--hash_bits', type=int, default=PVT2Config.HASH_BITS)
parser.add_argument('--test_type', type=str, default='normal')
parser.add_argument('--train_h', type=bool, default=True)
if __name__ == '__main__':
    args = parser.parse_args()
    type_ = args.type
    path_ = args.path
    h_path = args.pretrained
    hash_bits = args.hash_bits
    model_cls = args.model_cls
    test_cls = args.test_cls
    test_compresses = args.c
    device = torch.device(f"cuda:{args.local_rank}")
    PVT2Config.NUM_CLASSES, Dataset, src_path = choices[args.type]
    if 'normal' == args.test_type:
        normal_test()
    elif 'cross_db' == args.test_type:
        cross_test()
    elif 'robust' == args.test_type:
        robust_test()
