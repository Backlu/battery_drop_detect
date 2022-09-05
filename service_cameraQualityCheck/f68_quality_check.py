#!/usr/bin/env python
# coding: utf-8

import os, sys
import datetime, time
import cv2
import pandas as pd
import numpy as np
import json
from common.hikvision_api import get_rtsp_url, get_replay_url

def releaseobject(df, name_object_value):
    for i,r in df.iterrows():
        object_dict = json.loads(df[name_object_value][i])
        for key in object_dict.keys():
            df.loc[i,key]=object_dict[key]
    return df

def f68_camera_quality_check(conn, debug=False):
    #FIXME 移到F68專案folder
    img_list = []
    cols = ['mid','detect_time','detect_status']
    detect_df = pd.DataFrame(columns = cols)
    sql_r = "SELECT f.camera_name, f.mid, f.ip, q.object_value, q.time as qualicy_check_time \
            FROM f68_camera_list_map f JOIN \
              (SELECT t.mid, t.object_value, t.time\
               FROM camera_quality t\
               INNER JOIN (select mid, max(time) as quality_check_time from camera_quality GROUP BY mid) c\
               on t.mid = c.mid and t.time = c.quality_check_time) q\
            ON f.mid = q.mid"
    df = pd.read_sql(sql_r, conn)
    df_68 = releaseobject(df, 'object_value')
    df_68 = df_68.replace({'quality': {'pass': 'camera pass', 'tape_shift': 'camera shift'}})
    if debug:
        df_68=df_68.iloc[:1]
        
    img_list = []
    for idx in range(0, len(df_68)):
        ip = df_68.iloc[idx]['ip']
        mid = df_68.iloc[idx]['mid']
        vpath = f'rtsp://admin:a1234567@{ip}/h265/ch1/main/av_stream'
        vidcap = cv2.VideoCapture(vpath)
        success = True
        success, image = vidcap.read()
        detect_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
        if not success:
            detect_df = detect_df.append(pd.Series([mid,detect_time,'RTSP failed'], index=cols),ignore_index=True)
            continue
        img_list.append((cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mid))
        if (datetime.datetime.now() - df_68.iloc[idx]['qualicy_check_time']).total_seconds()/60 > 80:
            # TODO: abnormal 可能是 HIK fail，待確認
            detect_df = detect_df.append(pd.Series([mid,detect_time,'camera abnormal'], index=cols),ignore_index=True)
            continue
        status = df_68.iloc[idx]['quality']
        time = df_68.iloc[idx]['qualicy_check_time']
        detect_df = detect_df.append(pd.Series([mid,time,status], index=cols),ignore_index=True)
        
    return detect_df, img_list
        
