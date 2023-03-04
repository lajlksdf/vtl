import os

if __name__ == '__main__':
    path = r"D:\dataset\fake\fake"
    for pic_dir in os.listdir(path):
        video_dir = os.path.join(path, pic_dir)
        cmd = f'ffmpeg -f image2 -i {video_dir}/%05d.png {video_dir}.gif'
        os.system(cmd)
