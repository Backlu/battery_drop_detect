#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('.')
import os, time, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import pandas as pd
import tensorflow as tf
import multiprocessing
from service_batteryDropDetect.f45movement import F45Movement
import logging
from common.log import init_logging
from common.db import Database_Connection
from common.hikvision_api import get_video_stream
from common.utils import get_data_dir, get_model_dir, gpu_ram_config
MODEL_DIR = get_model_dir()
DATA_DIR = get_data_dir()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['ALIYUN_COM_GPU_MEM_IDX'] = '0'
os.environ['ALIYUN_POD_GPU_MEMORY'] = '1683'
os.environ['PRESERVE_GPU_MEMORY'] = '763'


#stream_mode Defalut 1 for RTSP mode
def f45_battDrop_detect(mid, crontabEndTimeOfHour, crontabEndTimeOfMin, is_online=True, stream_mode=1):
    if stream_mode == F45Movement.STEAM_MODE_REPLAY:
        print('warning! use replay mode')
        logging.warn('warning! use replay mode')
              
    rtsp_url = get_video_stream(stream_mode, mid)
    print(f'mid:{mid}, rtsp url:{rtsp_url}, stream mode:{stream_mode}')
    logging.info(f'mid:{mid}, rtsp url:{rtsp_url}, stream mode:{stream_mode}')
    
    start_time = datetime.datetime.now()
    end_time = start_time.replace(hour=crontabEndTimeOfHour, minute=crontabEndTimeOfMin, second=0)
    img_dir_draw, img_dir_raw = init_output_folder(mid)
    f45movement = F45Movement(camera_mid=mid, dir_draw=img_dir_draw, dir_raw=img_dir_raw, is_online=is_online, stream_mode=stream_mode)
    vidcap = cv2.VideoCapture(rtsp_url)
    success = True
    fid = 0
    
    while success:
        success, image = vidcap.read()
        if image is None:
            multiprocessing.Process(target = saveImageNone, args = (mid, fid)).start()
            print('last: None', mid, fid, datetime.datetime.today())
            vidcap = cv2.VideoCapture(rtsp_url)
            time.sleep(2)
            success, image = vidcap.read()
            if datetime.datetime.now() > end_time:
                print('time to break')
                logging.info('time to break')
            break
            
        if success is False:
            print('unexpect break')
            logging.info('unexpect break')
            break
            
        image = image[:608,:720]
        fid = fid + 1
        f45movement.detect(fid, image)
        
        if datetime.datetime.now() > end_time:
            print('time to break')
            logging.info('time to break')
            break
            
        if (stream_mode==F45Movement.STEAM_MODE_REPLAY) & (f45movement.detect_cnt > 5) :
            print('detect test pass, break')
            logging.info('detect test pass, break')
            break
        if (stream_mode==F45Movement.STEAM_MODE_TEST_FILE) & (f45movement.detect_black_cnt > 0) :
            print('detect black test pass, break')
            logging.info('detect black test pass, break')
            break
            
def saveImageNone(mid, fid):
    db = Database_Connection()
    rtsp_log = {'camera_name':mid, 'fid':fid, 'date_time':datetime.datetime.now(), 'type':'rtsp'}
    rtsp_df = pd.DataFrame(rtsp_log, index=[0])
    rtsp_df.to_sql('rtsp_log', con=db.engine, if_exists='append', index = False)
    

def init_output_folder(mid):
    dt= datetime.datetime.today()   
    dt_str = f'{dt:%m%d}'
    os.path.join(DATA_DIR, f'f45_output/{dt_str}')
    
    img_dir_dt = os.path.join(DATA_DIR, f'f45_output/{dt_str}')
    img_dir_draw = os.path.join(DATA_DIR, f'f45_output/{dt_str}/{mid}')
    img_dir_raw = os.path.join(DATA_DIR, f'f45_output/{dt_str}/{mid}/raw')
    
    if not os.path.exists(img_dir_dt):
        os.makedirs(img_dir_dt)
        os.chmod(img_dir_dt, 0o777)
    assert os.path.exists(img_dir_dt)
    
    if not os.path.exists(img_dir_draw):
        os.makedirs(img_dir_draw)
        os.chmod(img_dir_draw, 0o777)
    assert os.path.exists(img_dir_draw)

    if not os.path.exists(img_dir_raw):
        os.makedirs(img_dir_raw)
        os.chmod(img_dir_raw, 0o777)
    assert os.path.exists(img_dir_raw)
    
    return img_dir_draw, img_dir_raw        

if __name__ == '__main__':
    args = sys.argv[1:]
    crontabEndTimeOfHour = int(args[0])
    crontabEndTimeOfMin = int(args[1])
    camera_mid = args[2]
    init_logging('batteryDropDetect')
    gpu_ram_config()
    f45_battDrop_detect(camera_mid, crontabEndTimeOfHour, crontabEndTimeOfMin, is_online=True)
