import logging
import os
import time

logger = logging.getLogger()


def init_log():
    logger.setLevel(logging.INFO)  # Log等级总开关
    rq = time.strftime('%Y%m%d', time.localtime(time.time()))
    log_path = './logs/'
    logfile = log_path + rq + '.log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


init_log()
