# coding:utf-8

import logging
from logging.handlers import TimedRotatingFileHandler
import os, datetime
from common.utils import get_data_dir, get_model_dir, gpu_ram_config
DATA_DIR = get_data_dir()

def init_logging(filename_prefix='?'):
    LOG_PATH = os.path.join(f'{DATA_DIR}','log')
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    root = logging.getLogger()
    level = logging.INFO
    filename = f'{LOG_PATH}/{filename_prefix}_{datetime.datetime.now().strftime("%Y-%m-%d")}.log'
    format = '%(asctime)s %(levelname)s %(module)s.%(funcName)s Line:%(lineno)d %(message)s'
    #format = '%(asctime)s %(filename)s Line:%(lineno)d %(message)s'
    #filename, when to changefile, interval, backup count
    hdlr = TimedRotatingFileHandler(filename, "midnight", 1, 14)
    fmt = logging.Formatter(format)
    hdlr.setFormatter(fmt)
    root.addHandler(hdlr)
    root.setLevel(level)
    