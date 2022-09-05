#!/usr/bin/env python
# coding: utf-8

#copy raw images to 58, 12 for AI prediction recheck
#change to dest path to /mnt/hdd1/Data/f45movement/f45_output


import os, stat
import glob
from datetime import datetime, timedelta
import pandas as pd
import logging
from common.log import init_logging
from common.utils import get_data_dir
DATA_DIR = get_data_dir()

init_logging('copyImage')

start_time = datetime.now()
end_time = start_time.replace(hour=8, minute=0, second=0)
date = start_time - timedelta(days=1)
date = date.strftime("%m%d")

image_path1 = os.path.join(DATA_DIR, 'f45_output/{date}/*/*.jpg')
image_path2 = os.path.join(DATA_DIR, 'f45_output/{date}/*/*/*.jpg')
files = glob.glob(image_path1) + glob.glob(image_path2)
# scp fail data
for fpath in files:
    d_fpath = "/".join(fpath.split('/')[:-1])
    print(f"scp {fpath} elf@10.142.3.58:{d_fpath}")
    logging.info(f"scp {fpath} elf@10.142.3.58:{d_fpath}")
    os.system(f"scp {fpath} elf@10.142.3.58:{d_fpath}")
    print(f"scp {fpath} tpe-aa-01@10.109.6.12:{d_fpath}")
    logging.info(f"scp {fpath} tpe-aa-01@10.109.6.12:{d_fpath}")
    os.system(f"scp {fpath} tpe-aa-01@10.109.6.12:{d_fpath}")
    now_time = datetime.now()
    if now_time >  end_time:
        break