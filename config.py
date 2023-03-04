# hash bit length
from torchvision.transforms import transforms


class GlobalConfig:
    IS_DISTRIBUTION = False
    TRAIN_STEP = 30
    SET_PATH = '../h'
    TRAIN = 'train'
    TEST = 'test'
    OUT_CHANNELS = 3
    ALL_DIM = 192
    APP_DIR = './'
    NET_G = 'net_g.pth'
    NET_H = '_net_h.pth'


class PVT2Config(GlobalConfig):
    HASH_BITS = 512

    NUM_FRAMES = 4
    FRAMES_STEP = 1
    type = 'pvt2'
    BATCH_SIZE = 10
    base_lr = 1e-4
    train_h = True
    image_based = False
    IN_CHANNELS = 3
    NUM_CLASSES = 1000
    PATCH_SIZE = 7
    IMAGE_SIZE = 224

    # model config
    EMBED_DIMS = [64, 128, 192, 256]
    NUM_HEADS = [1, 2, 6, 8]
    MLP_RATIOS = [8, 8, 4, 4]
    QKV_BIAS = True
    QK_SCALE = None
    DROP_RATE = 0.1
    ATTN_DROP_RATE = 0.0
    DROP_PATH_RATE = 0.1
    DEPTHS = [3, 4, 8, 3]
    SR_RATIOS = [8, 4, 2, 1]
    NUM_STAGES = 4
    LINEAR = False

    mean = [0.447, 0.450, 0.417]
    std = [0.220, 0.220, 0.220]
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
