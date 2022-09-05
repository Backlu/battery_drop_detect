#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('.')
import pandas as pd
import datetime
from service_batteryDropDetect.schedule import inference_pod
from common.db import Database_Connection
from common.log import init_logging
from common.utils import gpu_ram_config
from service_batteryDropDetect.f45movement import F45Movement
from service_batteryDropDetect.f45_inference_main_online import f45_battDrop_detect


def test_battery_detect():
    db = Database_Connection()
    table_name = 'inference_schedule_setting'
    cameraID_df = pd.read_sql(f"select server_id, mid from {table_name}", db.engine)
    today = datetime.datetime.now().strftime("%Y%m%d")
    cam = cameraID_df.iloc[0]
    gpu_id = cam['server_id']
    camera_mid = cam['mid']

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['ALIYUN_COM_GPU_MEM_IDX'] = '0'
    os.environ['ALIYUN_POD_GPU_MEMORY'] = '1683'
    os.environ['PRESERVE_GPU_MEMORY'] = '763'

    init_logging('unit_test')
    #FIXME: OOM issue
    #gpu_ram_config()
    endtime = datetime.datetime.now()+ datetime.timedelta(minutes=15)
    f45_battDrop_detect(camera_mid, endtime.hour, endtime.minute, is_online=False, stream_mode=F45Movement.STEAM_MODE_REPLAY)

def test_battery_drop_detect():
    db = Database_Connection()
    table_name = 'inference_schedule_setting'
    cameraID_df = pd.read_sql(f"select server_id, mid from {table_name}", db.engine)
    today = datetime.datetime.now().strftime("%Y%m%d")
    cam = cameraID_df.iloc[0]
    gpu_id = cam['server_id']
    camera_mid = cam['mid']

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['ALIYUN_COM_GPU_MEM_IDX'] = '0'
    os.environ['ALIYUN_POD_GPU_MEMORY'] = '1683'
    os.environ['PRESERVE_GPU_MEMORY'] = '763'

    init_logging('unit_test')
    #FIXME: OOM issue
    #gpu_ram_config()
    endtime = datetime.datetime.now()+ datetime.timedelta(minutes=15)
    f45_battDrop_detect(camera_mid, endtime.hour, endtime.minute, is_online=False, stream_mode=F45Movement.STEAM_MODE_TEST_FILE)    
    