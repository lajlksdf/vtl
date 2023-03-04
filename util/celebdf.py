import os

if __name__ == '__main__':
    for j in range(0, 62):
        for i in range(0, 16):
            cmd = f'mkdir id{j}_{"%04d" % i}'
            os.system(cmd)
            cmd = f'mv id{j}_*_{"%04d" % i}.mp4 id{j}_{"%04d" % i}/'
            os.system(cmd)
