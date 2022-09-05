#!/usr/bin/env python
# coding: utf-8

# # F45 Hourly Check 
# Copyright © 2021 AA

# 前提
# - 海康時間 = 主機時間
# - 目前只有5L2

# History:
# 03/23: initial commit
# 05/04: f45_anomaly_info -> cnt; f45_anomaly_info2 -> unit

import sys
sys.path.append('.')
from sqlalchemy import Table, Column, String, Integer, MetaData, create_engine, insert, Text, DateTime, TIME
import pymysql
import time, datetime
import pandas as pd
import cv2
import threading, multiprocessing
import os
import numpy as np
from common.db import Database_Connection
import logging
from common.log import init_logging

def update_anomaly_sn_record(mid, err_dic, conn):
    multiprocessing.Process(target=to_database, args=(mid, conn, err_dic['error_record'], err_dic['error_type'], sn, err_dic['in_station_time'], err_dic['emp_no'], img_cnt, jpg_path)).start()
    
def to_database(mid, engine, error_time, error_type, serial_number, time, op_id, cnt, jpg_link):
    metadata = MetaData(bind=engine)
    anomaly_info = Table('f45_anomaly_sn', metadata,
                         Column('vid', String(50), primary_key=True),   
                         Column('last_error_time', String(50),  primary_key=True), 
                         Column('error_type', String(50)),
                         Column('serial_number', String(50)),
                         Column('in_station_time', String(50)),
                         Column('op_id', String(50)),
                         Column('number_of_errors', Integer),
                         Column('jpg_link', Text),
                        )
    metadata.create_all(engine)
    conn = engine.connect()
    act = insert(anomaly_info).values(vid=mid,
                                      last_error_time = error_time,
                                      error_type = error_type,
                                      serial_number = serial_number,
                                      in_station_time = time,
                                      op_id = op_id,
                                      number_of_errors = cnt,
                                      jpg_link = jpg_link,
                                     )
    conn.execute(act)
    conn.close()
        
def merge_img(imagelist):
    for n in range(1, len(imagelist)):
        if n <2:
            mergeimg = np.hstack((imagelist[0], imagelist[1]))
        else:
            mergeimg = np.hstack((mergeimg, imagelist[n])) 
    return mergeimg
     
def checkFolderExist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_batt_drop_log(conn, db_table):
    sql = "select * from f45_camera_list_map"
    df_camera = pd.read_sql(sql, conn)
    df_list = []
    for r in df_camera.itertuples():
        mid = r.mid
        err_time = (datetime.datetime.now() - datetime.timedelta(days = 1)).strftime("%Y-%m-%d %H:%M:%S")
        sql = f"select * from {db_table} \
               where vid = %(vid)s and error_time >= %(t)s and serial_number is NULL order by error_time ASC"
        batt_drop_df = pd.read_sql(sql, conn, params={'vid':mid, 't':err_time})
        batt_drop_df['mes_line'] = r.mes_line
        batt_drop_df['mes_station'] = r.mes_station
        
        df_list.append(batt_drop_df)
    batt_drop_df = pd.concat(df_list)
    
    return batt_drop_df

def get_mes_sn(line, station, error_time, conn):
    end = error_time.strftime("%Y-%m-%d %H:%M:%S")
    start = (error_time - datetime.timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")    
    sql_m = "select * from ipcamera.sn_detail_min where model_name like '%%W4%%' and line_name=%(lin)s and station_name=%(sta)s and in_station_time > %(t1)s and in_station_time <= %(t2)s order by in_station_time DESC limit 1"
    mes_sn_df = pd.read_sql(sql_m, conn, params={'lin':line, 'sta':station, 't1':start, 't2':end})
    has_sn = True if len(mes_sn_df)>0 else False
    sn_info = mes_sn_df.iloc[0] if has_sn else None
    return has_sn, sn_info

def save_recheck_image(mid, sn, image_list, img_dir, err_dic):
    img_cnt = len(image_list)
    if img_cnt <= 1:
        img = image_list[0]
    else:
        img = merge_img(image_list)
    checkFolderExist(img_dir)
    jpg_path = os.path.join(img_dir, f'{sn}.jpg')
    cv2.imwrite(jpg_path, img)
    
def update_batt_drop_record(sn, err_dic, conn, db_table):
    in_station_time = err_dic['in_station_time']
    emp_no = err_dic['emp_no']
    error_time = err_dic['error_time']
    print(f'sn:{sn}, error_time:{error_time}, emp_no:{emp_no}')
    logging.info(f'sn:{sn}, error_time:{error_time}, emp_no:{emp_no}')
    conn.execute(f"UPDATE {db_table} SET serial_number ='{sn}', in_station_time ='{in_station_time}', op_id ='{emp_no}' WHERE error_time='{error_time}';")    
    
def sync_AI_detection_and_MES_sn(is_online=True):
    if is_online:
        db_table = 'f45_anomaly_info'
    else:
        db_table = 'f45_anomaly_info_rtsp'    
    db = Database_Connection()
    batt_drop_df = get_batt_drop_log(db.engine, db_table)
    sn_record = ''
    image_list = []
    err_dic = {}
    for i, r in batt_drop_df.iterrows():
        line = r['mes_line']
        station = r['mes_station']    
        img_path = r['jpg_link']

        # FIXME: 正式上線環境不需要這段replace
        if '/mnt/hdd1/Data/f45movement' not in img_path:
            img_path = img_path.replace('/mnt/hdd1', '/mnt/hdd1/Data/f45movement')

        mid = r['vid']
        error_time = r['error_time']
        image= cv2.imread(img_path)
        if image is None: 
            continue
        has_sn, sn_info = get_mes_sn(line, station, error_time, db.engine)
        if not has_sn:
            continue
        serial_number = sn_info.serial_number
        if (serial_number != sn_record) & (i>0):
            img_dir = os.path.join(os.path.dirname(img_path), 'image')
            save_recheck_image(mid, sn_record, image_list, img_dir, err_dic)
            update_anomaly_sn_record(mid, err_dic, db.engine)
            image_list= []
            err_dic={}

        sn_record = serial_number        
        image_list.append(image)
        err_dic['in_station_time'] = sn_info.in_station_time.to_pydatetime()
        err_dic['error_record'] = r['error_time'].to_pydatetime()
        err_dic['error_type'] = r['error_type']
        err_dic['emp_no'] = sn_info.emp_no
        err_dic['error_time'] = error_time
        update_batt_drop_record(serial_number, err_dic, db.engine, db_table)
    

if __name__ == '__main__':
    args = sys.argv[1:]
    is_online = eval(args[0])
    init_logging('HourlyAIReport')
    sync_AI_detection_and_MES_sn(is_online)
    
