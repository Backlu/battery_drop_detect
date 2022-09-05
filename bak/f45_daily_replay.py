#!/usr/bin/env python
# coding: utf-8

# In[21]:

import os, sys, joblib, glob, shutil, json, time, datetime
import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gc
import requests
import imageio
from collections import Counter
from f45movement import F45Movement
from IPython.display import display, Markdown, Latex
import pymysql
from sqlalchemy import Table, Column, String, Integer, MetaData, create_engine, update, Float, DateTime, TIME

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus)>0:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print('No GPU')

# Load model
f45movement = F45Movement()    


# In[23]:


def getReplayUrl(cameraName, t1_str, t2_str):
    url = 'http://10.142.3.58:8081/v1/api/ipcamera/replayurl/name'
    data = {
        "cameraName":cameraName, 
        "beginTime":t1_str, 
        "endTime":t2_str,
        "expand":"streamform=rtp",        
#        "expand":"transcode=1&resolution=D1&bitrate=1024&framerate=15&streamform=rtp"
    }
    #print(f'{cameraName}: {t1_str} ~ {t2_str}')
    data_json = json.dumps(data)
    headers = {'Content-type': 'application/json'}

    response = requests.post(url, data=data_json, headers=headers)
    jsonObject = response.json()
#     print(jsonObject)
    replayUrl = ""
    if jsonObject['code'] == '200':
        replayUrl = jsonObject['result']['replayUrl']
    print(replayUrl)
    return replayUrl

def checkFolderExist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def checkImageNone(fid1, cameraName, cap, t1_str, t2_str):
    print('last none fid:', fid1, datetime.datetime.today())
    time.sleep(60)
    replayUrl = getReplayUrl(cameraName, t1_str, t2_str)
    cap = cv2.VideoCapture(replayUrl)
    ret, image = cap.read()
    return ret, image

# main

print('start')
engine = create_engine('mysql+pymysql://root:123456@10.142.3.58:3306/ipcamera?charset=utf8mb4')
sql_r = "SELECT * FROM f45_camera_list_map"
camera_df = pd.read_sql(sql_r, engine)
#camera_df = pd.read_excel('doc/camera_list_map.xlsx')
today = datetime.datetime.now()
yesterday = today - datetime.timedelta(days = 1)
date_str = yesterday.strftime("%m%d")
shifts = {'day': [datetime.time(9, 30), datetime.time(9, 35)], 
          'night':[datetime.time(21, 30), datetime.time(21, 35)]}

# 先把資料夾開好
for idx in range(0, len(camera_df)):
    camera_name = camera_df.iloc[idx]['camera_name']
    mid = camera_df.iloc[idx]['mid']
    for shift in shifts.keys():
        folder = f'/mnt/hdd1/share/{date_str}/{mid}/{shift}'
        checkFolderExist(folder)

for idx in range(0, len(camera_df)):
    camera_name = camera_df.iloc[idx]['camera_name']
    mid = camera_df.iloc[idx]['mid']
    for shift in shifts.keys():
        print(mid,shift,datetime.datetime.now())
        folder = f'/mnt/hdd1/share/{date_str}/{mid}/{shift}'
        # checkFolderExist(folder)
        t1 = datetime.datetime.combine(yesterday.date(), shifts[shift][0])
        t2 = datetime.datetime.combine(yesterday.date(), shifts[shift][1])
        start_dt = t1.strftime('%Y-%m-%dT%H:%M:%S.000+08:00') #start_time_pc的開始時間
        end_dt = t2.strftime( '%Y-%m-%dT%H:%M:%S.000+08:00') #start_time_pc的開始時間
        vpath = getReplayUrl(camera_name, start_dt, end_dt)
        vidcap = cv2.VideoCapture(vpath)
        success = True
        fid = 0
        while success:
            success, image = vidcap.read()
            if image is None:
                print(f'no image of {mid} {shift}')
                continue       
            fid = fid + 1
            image2 = image[:608,:720]
            detections = f45movement.yolo_detect(image2, f45movement.darknet_image, f45movement.net_wh)
            _, f_score = f45movement.yolo_bat(image2, detections)
            if (f_score > 0) or (fid % 1000 == 0):
                now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')[:-5]    
                jpg_path = os.path.join(folder, f'{mid}_{now_str}_{fid}.jpg')
                cv2.imwrite(jpg_path, image2)

# mid = 'F45_5L2'
# camera_df = pd.read_excel(f'doc/camera_list_map.xlsx')
# camera_name = camera_df[camera_df['mid']==mid]['camera_name'].values[0]
# ip = camera_df[camera_df['mid']==mid]['ip'].values[0]
# line = camera_df[camera_df['mid']==mid]['mes_line'].values[0]
# station = camera_df[camera_df['mid']==mid]['mes_station'].values[0]
# fps = 15

# shifts = ['day', 'night']
# today = datetime.datetime.now()
# #today = datetime.datetime.now()-datetime.timedelta(days=1)
# today_str = today.strftime("%m%d")

# folder = f'/mnt/hdd1/share/{today_str}'
# checkFolderExist(folder)

# for shift in shifts:
#     if shift == 'day':
#         t1 = datetime.datetime.combine(today.date(), datetime.time(10, 10))
#         t2 = datetime.datetime.combine(today.date(), datetime.time(10, 40))
#     else:
#         t1 = datetime.datetime.combine(today.date(), datetime.time(22, 10))
#         t2 = datetime.datetime.combine(today.date(), datetime.time(22, 40))
#     folder = f'/mnt/hdd1/share/{today_str}/{shift}'
#     checkFolderExist(folder)        
#     start_dt= t1.strftime('%Y-%m-%dT%H:%M:%S.000+08:00') #start_time_pc的開始時間
#     end_dt= t2.strftime( '%Y-%m-%dT%H:%M:%S.000+08:00') #start_time_pc的開始時間

#     vpath = getReplayUrl(camera_name, start_dt, end_dt)
#     vidcap = cv2.VideoCapture(vpath)
#     imagelist=[]
#     success = True
#     fid = 0
#     while success:
#         success, image = vidcap.read()
#         if image is None:
#             #success, image = checkImageNone(fid, mid, vidcap)
#             break       
#         fid = fid + 1
#         if fid % 3 != 0:
#             continue        
#         #imagelist.append(image)
#         now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')[:-5]
#         jpg_path = os.path.join(folder, f'{mid}_{shift}_{now}_{fid}.jpg')
#         cv2.imwrite(jpg_path, image)
        
# #     writer = imageio.get_writer(os.path.join(folder, f'{mid}_{start_dt}.mp4'), format='mp4', mode='I', fps=30)
# #     for img in imagelist:
# #         writer.append_data(img[:,:,::-1])
# #     writer.close()
# #     del imagelist
# #     gc.collect()

