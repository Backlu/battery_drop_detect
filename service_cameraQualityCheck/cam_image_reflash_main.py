#!/usr/bin/env python
# coding: utf-8

import sys, os
sys.path.append('.')
import pandas as pd
import cv2
from common.db import Database_Connection
from service_cameraQualityCheck.camera_image_plot import f45_camera_imgplot, f68_camera_imgplot
from common.hikvision_api import get_rtsp_url, get_replay_url, get_rtsp_url_f68
import logging
from common.log import init_logging
from common.utils import get_data_dir, gpu_ram_config, clean_folder
DATA_DIR = get_data_dir()

def reflash_f45_img(debug=False):
    db = Database_Connection()  
    sql_r = "SELECT * FROM f45_camera_list_map"
    camera_df = pd.read_sql(sql_r, db.engine)
    if debug:
        camera_df=camera_df.iloc[:1]    
    img_list = []
    for r in camera_df.itertuples():
        rtsp_url = get_rtsp_url(r.mid)
        vidcap = cv2.VideoCapture(rtsp_url)
        success, image = vidcap.read()
        if not success:
            continue
        img_list.append((cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mid))
    img_save_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_image_f45.jpg')        
    f45_camera_imgplot(camera_df['mid'].values, img_list, img_save_path, debug=debug)        

def reflash_f68_img(debug=False):
    db = Database_Connection()  
    sql_r = "SELECT * FROM f68_camera_list_map"
    camera_df = pd.read_sql(sql_r, db.engine)    
    if debug:
        camera_df=camera_df.iloc[:1]
    img_list = []        
    for r in camera_df.itertuples():
        rtsp_url = get_rtsp_url_f68(r.mid)
        vidcap = cv2.VideoCapture(rtsp_url)
        success, image = vidcap.read()
        if not success:
            continue
        img_list.append((cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mid))
    img_save_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_image_f68.jpg')        
    f68_camera_imgplot(camera_df['mid'].values, img_list, img_save_path, debug=debug)        
    

if __name__ == "__main__":
    init_logging('CamQualityCheck')
    output_dir = os.path.join(DATA_DIR, 'camera_quality_check')
    clean_folder(output_dir)    
    reflash_f45_img()
    reflash_f68_img()
    print('camera check complete')
    logging.info('camera check complete')    
    
