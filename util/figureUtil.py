import os

import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor

from config import PVT2Config
from layer.helper import to_image

x_ = [0, 1.1, 1.8, 3.1, 4.0]  # 数据点


def draw_figure(title, p1, p2: [], legends: [], name='Draw.jpg', x='x', y='y', markers=None, fontsize=17):
    plt.figure('Draw')
    plt.xlabel(x, fontdict={'size': fontsize})
    plt.ylabel(y, fontdict={'size': fontsize})
    # plt.title(title, fontdict={'size': fontsize})
    plt.xticks(fontsize=fontsize - 6)
    plt.yticks(fontsize=fontsize - 6)
    plt.rcParams.update({'font.size': fontsize - 6})
    for i in range(len(p2)):
        m = markers[i] if markers else 'o'
        p = p2[i]
        plt.plot(p1, p, marker=m, markersize=3)
    plt.legend(legends)
    plt.draw()
    plt.savefig(name)
    plt.close()


def draw3d(x, y, z, title, name):
    plt.title(title)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    plt.draw()
    plt.pause(10)
    plt.savefig(name)
    plt.close()


def analyze_hash_dist(name, itr_times, hash_dists):
    plt.title('Hash Analysis')
    plt.plot(itr_times, hash_dists, color='black', label='hash distance')
    plt.legend()

    plt.xlabel('iteration times')
    plt.ylabel('value')
    plt.draw()
    plt.savefig(name)  # 保存图象
    plt.close()


def merge_pic(files: [], g_tensor: Tensor, mask: Tensor, name):
    multi = 3
    b, t, c, h, w = g_tensor.shape
    g_tensor = g_tensor.detach().cpu()
    images = Image.new('RGB', (w * t, h * b * multi))
    for j in range(b * multi):
        for i in range(t):
            if j % multi == 0:
                img = to_image(g_tensor[j // multi][i])
            elif j % multi == 1:
                img = to_image(mask[j // multi][i])
            else:
                img = Image.open(files[i][j // multi])
                img = img.resize((PVT2Config.IMAGE_SIZE, PVT2Config.IMAGE_SIZE))
            images.paste(img, box=(i * w, j * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def merge_images(srcs: [], files: [], masks: [], outs: Tensor, name):
    multi = 4
    b, t, c, h, w = outs.shape
    images = Image.new('RGB', (w * t, h * b * multi))
    for j in range(b * multi):
        for i in range(t):
            if j % multi == 0:
                img = Image.open(srcs[i][j // multi])
                img = img.resize((PVT2Config.IMAGE_SIZE, PVT2Config.IMAGE_SIZE))
            elif j % multi == 1:
                img = Image.open(files[i][j // multi])
                img = img.resize((PVT2Config.IMAGE_SIZE, PVT2Config.IMAGE_SIZE))
            elif j % multi == 2:
                img = Image.open(masks[i][j // multi])
                img = img.resize((PVT2Config.IMAGE_SIZE, PVT2Config.IMAGE_SIZE))
            else:
                img = to_image(outs[j // multi][i])
            images.paste(img, box=(i * w, j * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def human_dataset(set_path=r'/home/dell/soft/ppf/vrf/train'):
    items = sorted(os.listdir(set_path))
    images = None
    for i in range(121):
        item = items[i]
        item_path = os.path.join(set_path, item)
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            img = sorted(os.listdir(src_dir))[0]
            img = Image.open(os.path.join(src_dir, img))
            if images is None:
                images = Image.new("RGB", (224 * 11, 224 * 11))
            img = img.resize((224, 224))
            images.paste(img, box=((i % 11) * 224, (i // 11) * 224))
    images.save('human.jpg')


def inpainting_dataset(set_path=r'E:\dataset\in\test\fake\CPNET'):
    items = sorted(os.listdir(set_path))
    images = None
    for i in range(49):
        item = items[i]
        item_path = os.path.join(set_path, item)
        if os.path.isdir(item_path):
            img = sorted(os.listdir(item_path))[0]
            img = Image.open(os.path.join(item_path, img))
            if images is None:
                images = Image.new("RGB", (224 * 7, 224 * 7))
            img = img.resize((224, 224))
            images.paste(img, box=((i // 7) * 224, (i % 7) * 224))
    images.save('inpainting.jpg')


def splicing_dataset(set_path=r'D:\dataset\splicing-tl\test\fake'):
    items = sorted(os.listdir(set_path))
    rows = 6
    images = Image.new('RGB', (224 * (rows - 1), 224 * rows))
    for i in range(30):
        item = items[i]
        item_path = os.path.join(set_path, item)
        if os.path.isdir(item_path):
            img = sorted(os.listdir(item_path))[0]
            img = Image.open(os.path.join(item_path, img))
            img = img.resize((224, 224))
            images.paste(img, box=((i // rows) * 224, (i % rows) * 224))
    images.save('splicing.jpg')


def draw_hash_act():
    tahn_itrs = [
        273.44444, 251.08424, 253.45017, 252.86706, 253.13611, 254.23905, 254.96861, 255.16560, 255.33178, 255.48414,
        255.47260, 255.65378, 255.74449,
        255.81153, 255.82911, 255.91406, 255.87447, 255.86140, 255.94848, 255.90836, 256.01729, 256.03104, 255.98020,
        256.03403, 256.05150, 256.05150,
        255.97429, 255.96048, 255.98877, 255.95074, 256.00887, 256.04158, 256.04119, 256.03052, 256.00739, 256.00287
    ]
    relu_itrs = [
        271.86667, 254.10238, 254.45638, 254.35606, 254.45108, 254.31435, 254.31938,
        254.84375, 255.28274, 255.56761, 255.70276, 255.77163, 255.84407, 255.79598, 255.93246, 255.93026, 255.94028,
        256.04851, 255.98798, 256.05427, 256.03975,
        255.98509, 255.97211, 256.05157, 255.99364, 256.06273, 256.06169, 256.06995, 256.08041, 256.08150, 256.04939,
        256.04913, 256.06945, 256.07180, 256.06907,
        256.05321
    ]
    sigmoid_itrs = [
        271.86667, 254.06529, 254.36193, 254.50459, 254.62315, 254.47905, 254.59380, 254.93542, 255.30138, 255.62282,
        255.77359, 255.84067, 255.89578, 255.81968,
        255.96296, 255.93224, 255.93495, 256.04265, 256.04265, 256.04265, 256.01639, 255.94931, 255.97147, 256.03200,
        255.96829, 256.01726, 256.00473, 256.03579,
        256.06370, 256.06593, 256.05610, 256.04663, 256.07969, 256.05224, 256.06612, 256.05805
    ]
    dises = []
    for i in range(len(tahn_itrs)):
        dises.append(i * 100)

    params = [tahn_itrs, relu_itrs, sigmoid_itrs]
    legends = ['Tahn', 'Relu', 'Sigmoid']
    markers = ['^', '|', 'v']
    draw_figure('Average Hamming distance of different activation functions', dises, params, legends,
                x='Training Iterations', y='Average Hamming distance', name='hash.png', markers=markers)


def draw_hash_loss():
    intra_itrs = [
        237.86842, 230.14298, 247.34240, 247.62318, 240.62167, 245.47375, 247.68307, 241.96013, 246.04352, 241.29938,
        245.27706, 241.10891, 231.82912, 243.13465,
        242.04785, 241.09979, 250.10139, 246.13876, 243.16564, 244.55183, 246.00137, 245.69082, 239.89200, 247.14058,
        235.91183, 240.31351, 246.49077, 239.28731,
        244.27341, 241.63568, 244.33105, 231.74277, 247.42424, 251.90453, 248.32809
    ]
    inter_itrs = [
        261.19928, 253.22575, 253.01024, 254.04848, 253.89128, 254.56061, 253.81030, 253.66237, 254.59895, 255.05935,
        254.93551, 254.92966, 255.35412, 255.52071,
        255.31497, 255.47105, 255.38815, 255.22793, 255.39448, 255.57138, 255.46683, 255.66395, 255.61268, 255.83782,
        255.93569, 255.91313, 255.84153, 255.62133,
        255.75111, 255.91388, 255.99641, 255.99867, 255.94497, 255.89012, 255.94063
    ]
    dises = []
    for i in range(len(intra_itrs)):
        dises.append(i * 100)

    params = [intra_itrs, inter_itrs]
    legends = ['Intra-Loss', 'Inter-Loss']
    draw_figure('Average Hamming distance with only one loss', dises, params, legends, x='Training Iterations',
                y='Average Hamming distance', name='hash_loss.png')


def analyze_acc():
    legends = ['FF++ Raw', 'FF++ C23', 'FF++ C40']
    raw = [0.852, 0.932, 0.948, 0.998, 0.991]
    c23 = [0.847, 0.944, 0.944, 0.998, 0.990]
    c40 = [0.846, 0.941, 0.946, 0.997, 0.991]

    params = [raw, c23, c40]
    dises = ['64bits', '128bits', '256bits', '512bits', '1024bits']
    draw_figure('Tracing accuracy of different hash bits', dises, params, legends, x='', y='Accuracy',
                name='compresses.png')


def analyze_data():
    legends = ['None', 'Detail', 'Noise', 'Blur', 'Median', 'Crop']
    Original = [0.852, 0.932, 0.948, 0.998, 0.991]
    Detail = [0.850, 0.930, 0.949, 0.999, 0.991]
    Gaussian = [0.844, 0.937, 0.944, 0.999, 0.990]
    Blur = [0.846, 0.934, 0.947, 0.998, 0.991]
    Median = [0.850, 0.935, 0.948, 0.998, 0.991]
    Cropping = [0.633, 0.801, 0.862, 0.983, 0.963]

    params = [Original, Detail, Gaussian, Blur, Median, Cropping]
    dises = ['64bits', '128bits', '256bits', '512bits', '1024bits']
    draw_figure('Tracing accuracy of different image processing', dises, params, legends, x='', y='Accuracy',
                name='data.png')


if __name__ == '__main__':
    analyze_data()
    analyze_acc()