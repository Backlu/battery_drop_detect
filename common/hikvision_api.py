#!/usr/bin/env python
# coding: utf-8

import json
import requests
import pandas as pd
from common.db import Database_Connection
import datetime

def get_video_stream(stream_mode, mid):
    #2 for replay
    if stream_mode == 2:
        t1 = datetime.datetime(2022,7,1,9,18,30)
        t2 = datetime.datetime(2022,7,1,9,30,0)
        start_dt = t1.strftime('%Y-%m-%dT%H:%M:%S.000+08:00') 
        end_dt = t2.strftime( '%Y-%m-%dT%H:%M:%S.000+08:00') 
        stream_url = get_replay_url(mid, start_dt, end_dt)
    elif stream_mode == 3:
        stream_url = '/mnt/hdd1/Data/f45movement/test_data/black_test.mp4'
    else:
        stream_url = get_rtsp_url(mid)
    return stream_url

def get_rtsp_url_f68(mid):
    #FIXME: 移回F68專案
    db = Database_Connection()
    camera_df = pd.read_sql(f'select * from f68_camera_list_map', db.engine)
    camera_name = camera_df[camera_df['mid']==mid]['camera_name'].values[0]
    ip = camera_df[camera_df['mid']==mid]['ip'].values[0]
    line = camera_df[camera_df['mid']==mid]['mes_line'].values[0]
    station = camera_df[camera_df['mid']==mid]['mes_station'].values[0]
    vpath = "rtsp://admin:a1234567@"+ ip +"/h265/ch1/main/av_stream"
    return vpath

def get_rtsp_url(mid):
    db = Database_Connection()
    camera_df = pd.read_sql(f'select * from f45_camera_list_map', db.engine)
    camera_name = camera_df[camera_df['mid']==mid]['camera_name'].values[0]
    ip = camera_df[camera_df['mid']==mid]['ip'].values[0]
    line = camera_df[camera_df['mid']==mid]['mes_line'].values[0]
    station = camera_df[camera_df['mid']==mid]['mes_station'].values[0]
    vpath = "rtsp://admin:a1234567@"+ ip +"/h265/ch1/main/av_stream"
    return vpath


def get_preview_url(mid):
    db = Database_Connection()
    camera_df = pd.read_sql(f'select * from f45_camera_list_map', db.engine)
    camera_name = camera_df[camera_df['mid']==mid]['camera_name'].values[0]
    
    url = 'http://10.142.3.58:8081/v1/api/ipcamera/previewurl/name'
    data = {
        "cameraName":camera_name,
        "expand":"streamform=rtp" ,
        #"transcode=1&resolution=D1&bitrate=512&framerate=15&streamform=rtp&snapshot=1"
    }
    data_json = json.dumps(data)
    headers = {'Content-type': 'application/json'}

    response = requests.post(url, data=data_json, headers=headers)
    jsonObject = response.json()
    replayUrl = None
    if jsonObject['code'] == '200':
        replayUrl = jsonObject['result']['replayUrl']
    #print(replayUrl)
    return replayUrl


def get_replay_url(mid, t1_str, t2_str):
    db = Database_Connection()
    camera_df = pd.read_sql(f'select * from f45_camera_list_map', db.engine)
    camera_name = camera_df[camera_df['mid']==mid]['camera_name'].values[0]
    
    url = 'http://10.142.3.58:8081/v1/api/ipcamera/replayurl/name'
    data = {
        "cameraName":camera_name, 
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
    #print(jsonObject)
    replayUrl = None
    if jsonObject['code'] == '200':
        replayUrl = jsonObject['result']['replayUrl']
    #print(replayUrl)
    return replayUrl
