#!/usr/bin/env python
# coding: utf-8

import os, sys
import datetime, time
import cv2
import pandas as pd
import numpy as np
from service_batteryDropDetect.f45movement import F45Movement
import logging
from common.hikvision_api import get_video_stream


def f45_camera_quality_check(f45movement, conn, debug=False):
    camera_df = get_f45_camera_list(conn)
    stream_mode = F45Movement.STEAM_MODE_REPLAY
    
    if stream_mode == F45Movement.STEAM_MODE_REPLAY:
        print('warning! use replay mode')
        logging.warn('warning! use replay mode')    
    
    mid_list = []
    detect_time_list = []
    detect_status_list = []
    detect_img_list = []
    for r in camera_df.itertuples():
        if debug:
            if r.mid !='F45_5L1':
                continue
        rtsp_url = get_video_stream(stream_mode, r.mid)
        if rtsp_url is None:
            print(f'skip {r.mid}, rtsp_url None')
            logging.warn(f'skip {r.mid}, rtsp_url None')
            continue
        wz_status, sample_img, log_dict = camera_quality_check(rtsp_url, f45movement)
        mid_list.append(r.mid)
        detect_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
        detect_time_list.append(detect_time)
        detect_status_list.append(wz_status)
        detect_img_list.append(sample_img)
        logging.info(f'mid:{r.mid}, status:{wz_status}, detectTime:{detect_time}, log:{log_dict}')
    cam_quality_df = pd.DataFrame({'mid':mid_list, 'detect_time':detect_time_list, 'detect_status':detect_status_list})
    return cam_quality_df, detect_img_list
    

def get_f45_camera_list(conn):
    sql_r = "SELECT * FROM f45_camera_list_map"
    camera_df = pd.read_sql(sql_r, conn)
    return camera_df        

def camera_quality_check(rtsp_url, f45movement):
    NUM_FRAMES = 120
    log_dict={}
    ret_dict={}
    ret_dict['ret_rtsp_sterm'] = True
    ret_dict['ret_fps'] = False
    ret_dict['ret_image_size'] = False
    ret_dict['ret_workzone'] = False
    
    # Streaming test
    img_list = []    
    vidcap = cv2.VideoCapture(rtsp_url)
    start_time = time.time()
    for fid in range(NUM_FRAMES):
        success, image = vidcap.read()
        if success==False:
            ret_dict['ret_rtsp_sterm']=False
            log_dict['ret_rtsp_sterm']=False
            break
        img_list.append(image)

    # FPS test
    if ret_dict['ret_rtsp_sterm']:
        fps  = NUM_FRAMES/(time.time()-start_time)
        ret_dict['ret_fps'] = True if fps>10 else False
        log_dict['fps']=fps

    # Image size test
    if ret_dict['ret_rtsp_sterm'] & ret_dict['ret_fps']:
        img_height, img_width = image.shape[:2]
        ret_dict['ret_image_size'] = False if ((img_width!=1280) or (img_height!=720)) else True
        log_dict['img_size(h,w)']=(img_height, img_width)

    # 工作區 test
    if ret_dict['ret_rtsp_sterm'] & ret_dict['ret_fps'] & ret_dict['ret_image_size']:
        ret_wz, wz_status,  wz_score, wz_dist = work_zone_test(img_list, f45movement)
        ret_dict['ret_workzone']= ret_wz
        ret_dict['wz_status']=log_dict['wz_status'] = wz_status
        log_dict['wz_socre'], log_dict['wz_dist']= wz_score, wz_dist

    wz_status = format_wz_status_message(ret_dict)
    sample_img = img_list[-1] if len(img_list)>0 else np.zeros([720, 1280, 3],dtype=np.uint8)
    
    return wz_status, sample_img, log_dict


def get_distance(loc1, loc2):
    dist = np.sqrt( (loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2 )
    return int(dist)

def work_zone_test(img_list, f45movement):
    WZ_SCORE_CRITERION = 20
    WZ_DIST_CRITERION = 100
    WZ_IMG_CROP_Y = 608
    WZ_IMG_CROP_X = 720
    wz_pass = False
    wz_status = 'camera abnormal'
    for img in img_list:
        img_crop = img[:WZ_IMG_CROP_Y,:WZ_IMG_CROP_X]
        wz_score, (wz_x, wz_y) = f45movement.detect_workzone(img_crop)
        wz_dist = get_distance((530, 300), (wz_x, wz_y))
        if wz_score>WZ_SCORE_CRITERION:
            wz_pass = True if wz_dist < WZ_DIST_CRITERION else False
            wz_status = 'camera pass' if wz_pass else 'camera shift'
            break
    return wz_pass, wz_status, wz_score, wz_dist

def format_wz_status_message(ret_dict):
    wz_status = ''
    if ret_dict['ret_rtsp_sterm']==False:
        wz_status = 'RTSP failed'
    elif ret_dict['ret_fps']==False:
        wz_status = 'fps too low'
    elif ret_dict['ret_image_size']==False:
        wz_status = 'image size error'
    elif ret_dict['ret_workzone']==False:
        wz_status = ret_dict['wz_status']
    else:
        wz_status = 'camera pass'
    return wz_status
