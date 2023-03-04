import os
import sys

if __name__ == '__main__':
    path = sys.argv[1]
    for f in os.listdir(path):
        if f.endswith('.mp4'):
            id = f.split('_')[0]
            name = f.replace(id, '').replace('.mp4', '')
            dir_name = os.path.basename(f).split('.')[0]
            cmd = f'mkdir {dir_name}'
            os.system(cmd)
            cmd = f'mv {id}_*{name}* {dir_name}'
            os.system(cmd)
