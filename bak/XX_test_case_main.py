#!/usr/bin/env python
# coding: utf-8

import warnings, os, cv2, joblib, datetime, logging
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
logging.basicConfig(level="ERROR")

import sys, glob, shutil, json, time, pytesseract
import pandas as pd
import tensorflow as tf
import gc
import numpy as np
from collections import Counter
from f45movement import F45Movement
import requests
import pymysql
from sqlalchemy import create_engine

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

def test_pass_case_0():
    
    # Load model
    f45movement = F45Movement()
    f45movement.vid = '5L2'

    vpath = '/mnt/hdd1/ipcamera_case_data/F45_5L2_2021-01-28T10:12:18.000+08:00_134.mp4'

    dt= datetime.datetime.today()   
    dt_str = f'{dt:%m%d}'
    folder = f'/mnt/hdd1/f45_output/{dt_str}'
    folder2 = f'/mnt/hdd1/f45_output/{dt_str}/raw'
    if not os.path.exists(folder):
        os.makedirs(folder)
    assert os.path.exists(folder)

#     if not os.path.exists(folder2):
#         os.makedirs(folder2)
#     assert os.path.exists(folder2)

    # Inference
    vidcap = cv2.VideoCapture(vpath)
    success = True
    fid = 0
    bf_record = 0
    bk_record = 0
    t_record = datetime.datetime.now()
    save_image = None
    save_image2 = None
    inference = True

    while success:
        success, image = vidcap.read()
        if image is None:
            break

        image = image[:608,:720]
        fid = fid + 1
        bf_record, bk_record, t_record, save_image, save_image2, inference = \
        f45movement.detect(fid, image, bf_record, bk_record, t_record, save_image, 
                           save_image2, folder, folder, inference)
            
        if fid > 140:
            break
            
    ret = save_image is not None
    print('test result', ret)   
    assert ret   
