import os
import glob
import shutil

# 指定目录路径
directory = './DFTL/test'

dest_dir = './dftl'


# 遍历匹配到的文件列表
def jpg_list(dir_path):
    jpg_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))

    return [jpg_files[0], jpg_files[len(jpg_files) // 2], jpg_files[-1]]


if __name__ == '__main__':
    for label in os.listdir(directory):
        old_dir = os.path.join(directory, label)
        images = jpg_list(old_dir)
        new_dir = os.path.join(dest_dir, label)
        if not os.path.exists(new_dir):
            # 如果目录不存在，则创建它
            os.makedirs(new_dir)
        for f in images:
            shutil.copy(f, new_dir)

