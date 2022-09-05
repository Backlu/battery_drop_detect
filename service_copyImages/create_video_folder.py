#!/usr/bin/env python
# coding: utf-8

#Create folder to save recheck images

import os, stat
import glob
from datetime import datetime, timedelta
import pandas as pd
from schedule import Database_Connection
import logging
from common.log import init_logging
from common.utils import get_data_dir
DATA_DIR = get_data_dir()

init_logging('create_video_folder')

output_path = os.path.join(DATA_DIR, 'f45_output')
db = Database_Connection()
cameras = pd.read_sql(f'select * from f45_camera_list_map', db.engine)
mids = cameras['mid'].tolist()
dt = datetime.now()
for mid in mids:
    d_folder = f'{output_path}/{dt:%m%d}/{mid}/'
    print(f'create folder:{d_folder} start')
    logging.info(f'create folder:{d_folder} start')
    if not os.path.exists(d_folder):
        os.makedirs(d_folder)
    os.chmod(output_path, 0o777)
    raw_folder = f'{output_path}/{dt:%m%d}/{mid}/raw'
    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder)
    os.chmod(output_path, 0o777)
    image_folder = f'{output_path}/{dt:%m%d}/{mid}/image'
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    os.chmod(output_path, 0o777)
    print(f'create folder:{d_folder} done')
    logging.info(f'create folder:{d_folder} done')    
