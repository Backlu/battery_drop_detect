#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('.')
import pandas as pd
from datetime import datetime
import numpy as np
from IPython.display import display, Markdown, Latex
from sqlalchemy import create_engine
import configparser
import pymysql.cursors
import multiprocessing
import os
import logging
from common.db import Database_Connection
from common.utils import get_config_dir
CONFIG_PATH = get_config_dir()
#import inspect



def inference_pod(gpu_id, mid, job_name):
    logging.info(f'bash sh/pod-f45.sh {gpu_id} {mid} {job_name}')
    stdout = os.system(f'bash sh/pod-f45.sh {gpu_id} {mid} {job_name}')
    
if __name__ == '__main__':
    args = sys.argv[1:]
    is_online = eval(args[0])
    
    init_logging('batteryDropDetect')
    db = Database_Connection()
    table_name = 'inference_schedule_setting'
    sd_df = pd.read_sql(f"select server_id, mid from {table_name}", db.engine)
    today = datetime.now().strftime("%Y%m%d")

    if is_online==False:
        logging.info('develop test mode')
        #只起一個k8s job測試
        cameraID_df = cameraID_df[cameraID_df['mid']=='F45_5L10']
        
    processes = []
    process_id = 0
    for index, row in sd_df.iterrows():
        gpu_id = row['server_id']
        mid = row['mid']
        job_name = row['mid'].replace('_', '-').lower() + '-' + today
        if 'F68' in mid:
            continue
        processes.append(multiprocessing.Process(target = inference_pod, args = (gpu_id, mid, job_name)))
        processes[process_id].start()
        process_id += 1
