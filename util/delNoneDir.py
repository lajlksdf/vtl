import os
import sys

if __name__ == '__main__':
    path = sys.argv[1]
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if len(os.listdir(dir)) == 0:
                print(dir)
                # cmd = f'rm -rf {dir}'
                # os.system(cmd)
