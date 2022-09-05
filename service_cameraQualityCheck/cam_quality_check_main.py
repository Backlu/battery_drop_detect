#!/usr/bin/env python
# coding: utf-8

import sys, os
sys.path.append('.')
import pandas as pd
from service_batteryDropDetect.f45movement import F45Movement
from common.db import Database_Connection
from service_cameraQualityCheck.f45_quality_check import f45_camera_quality_check
from service_cameraQualityCheck.f68_quality_check import f68_camera_quality_check
from service_cameraQualityCheck.camera_image_plot import f45_camera_imgplot, f68_camera_imgplot
import logging
from common.log import init_logging
from common.utils import get_data_dir, gpu_ram_config, clean_folder
DATA_DIR = get_data_dir()

def f45_cam_check(debug=False):
    f45movement = F45Movement()
    db = Database_Connection()
    cam_quality_df, detect_img_list = f45_camera_quality_check(f45movement, db.engine, debug=debug)
    img_save_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_image_f45.jpg')
    f45_camera_imgplot(cam_quality_df['mid'].values, detect_img_list, img_save_path, debug=debug)
    staus_save_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_status_f45.csv')
    cam_quality_df.to_csv(staus_save_path, index = False)
    if debug==False:
        update_quality_check_result_to_database(cam_quality_df, db.engine)
    
def f68_cam_check(debug=False):
    db = Database_Connection()
    cam_quality_df, detect_img_list = f68_camera_quality_check(db.engine, debug=debug)
    img_save_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_image_f68.jpg')
    f68_camera_imgplot(cam_quality_df['mid'].values, detect_img_list, img_save_path, debug=debug)
    staus_save_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_status_f68.csv')
    cam_quality_df.to_csv(staus_save_path, index=False)
    if debug==False:
        update_quality_check_result_to_database(cam_quality_df, db.engine)

    
def update_quality_check_result_to_database(cam_quality_df, conn):
    sql_r = "select * from inference_schedule_setting"
    df_inference_cam = pd.read_sql(sql_r, conn)
    df_inference_cam = df_inference_cam[df_inference_cam['mid'].isin(cam_quality_df['mid'])]
    for r in df_inference_cam.itertuples():
        cam_quality = cam_quality_df[cam_quality_df['mid']==r.mid].iloc[0]
        detect_status = cam_quality['detect_status']
        detect_time = cam_quality['detect_time']
        sql = f"UPDATE inference_schedule_setting SET quality_check_result ='{detect_status}', quality_check_time ='{detect_time}' WHERE mid='{r.mid}';"
        conn.execute(sql)
    
    
if __name__ == "__main__":
    init_logging('CamQualityCheck')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpu_ram_config(gpu_id=0, ram=3000)
    output_dir = os.path.join(DATA_DIR, 'camera_quality_check')
    clean_folder(output_dir)
    f45_cam_check()
    f68_cam_check()
    #下面這句不能刪掉, UI執行時會判斷是否出現這個句子
    print('camera check complete')
    logging.info('camera check complete')
