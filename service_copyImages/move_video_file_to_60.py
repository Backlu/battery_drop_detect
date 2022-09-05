#!/usr/bin/env python
# coding: utf-8

#copy raw images to 60 for CVAT annotation
#CVAT need to mount the folder: /mnt/hdd1/Data/f45movement/f45_output

import os, stat
import glob
from datetime import datetime, timedelta
import pandas as pd
import logging
from common.log import init_logging
from common.utils import get_data_dir
DATA_DIR = get_data_dir()

init_logging('copyImage')

today = datetime.now()
yesterday = today - timedelta(days = 1)
date = yesterday.strftime("%m%d")
file_path = os.path.join(DATA_DIR, f'f45_output/{date}/*/raw/*.jpg')
files = glob.glob(file_path)

# scp fail data 
for fpath in files:
    fhour = int(fpath.split('/')[-1].split('-')[3])
    mid = fpath.split('/')[-3]
    if (fhour < 8) or (fhour >19):
        d_fpath = os.path.join(DATA_DIR, f'f45_output/{date}/{mid}/night')
        #d_fpath = f'/mnt/hdd1/share/{date}/{mid}/night'
    else:
        d_fpath = os.path.join(DATA_DIR, f'f45_output/{date}/{mid}/day')
        #d_fpath = f'/mnt/hdd1/share/{date}/{mid}/day'
    print(f"scp {fpath} elf@10.142.3.60:{d_fpath}")
    logging.info(f"scp {fpath} elf@10.142.3.60:{d_fpath}")
    os.system(f"scp {fpath} elf@10.142.3.60:{d_fpath}")

