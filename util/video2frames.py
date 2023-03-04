import os
import sys

if __name__ == '__main__':
    path = sys.argv[1]
    output = sys.argv[2]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mp4'):
                video = os.path.join(root, file)
                out = os.path.join(output, file)
                os.makedirs(out)
                cmd = f'ffmpeg -i {video} -f image2 {out}/%5d.png'
                os.system(cmd)
