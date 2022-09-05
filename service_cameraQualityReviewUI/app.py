#!/usr/bin/env python
# coding: utf-8

## History:
# 3/3: 1st ver 
import sys
sys.path.append('.')
import os
import pandas as pd
import streamlit as st
import time
from paramiko import SSHConfig, SSHClient
import paramiko
from contextlib import closing
import numpy as np
import cv2
from sqlalchemy import create_engine
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import json
import glob
import random
from common.db import Database_Connection
from common.utils import get_data_dir, get_project_root
DATA_DIR = get_data_dir()
ROOT_DIR = get_project_root()

#Config
username = "elf"
password = "note123@"
hostname = "10.142.3.60"
port = 22

def get_camera_status():
    camera_status_f45_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_status_f45.csv')
    camera_status_f68_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_status_f68.csv')
    df_f45 = pd.read_csv(camera_status_f45_path)
    df_f68 = pd.read_csv(camera_status_f68_path)
    df = pd.concat([df_f45,df_f68])
    return df
    
#@st.cache(ttl=300)
def get_camera_status_ssh():
    camera_status_f45_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_status_f45.csv')
    camera_status_f68_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_status_f68.csv')
    with closing(SSHClient()) as client:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port, username, password)
        with closing(client.open_sftp()) as sftp:
            with sftp.open(camera_status_f45_path) as f:
                df_f45 = pd.read_csv(f)
            with sftp.open(camera_status_f68_path) as f:
                df_f68 = pd.read_csv(f)
            df = pd.concat([df_f45,df_f68])
    return df

def get_camera_image():
    camera_image_f45_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_image_f45.jpg')
    camera_image_f68_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_image_f68.jpg')
    img_f45 = cv2.imread(camera_image_f45_path)
    img_f68 = cv2.imread(camera_image_f68_path)
    return img_f45, img_f68
    
#@st.cache(ttl=300)
def get_camera_image_ssh():
    camera_image_f45_path_OLD = f'/mnt/hdd1/QOO/doc/camera_image.jpg'
    camera_image_f68_path_OLD = f'/mnt/hdd1/QOO/doc/camera_image_f68.jpg'    
    camera_image_f45_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_image_f45.jpg')
    camera_image_f68_path = os.path.join(DATA_DIR, 'camera_quality_check/camera_image_f68.jpg')
    with closing(SSHClient()) as client:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port, username, password)
        with closing(client.open_sftp()) as sftp:
            try:
                with sftp.open(camera_image_f45_path) as f:
                    img_f45 = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)
                with sftp.open(camera_image_f68_path) as f:
                    img_f68 = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)
            except:
                with sftp.open(camera_image_f45_path_OLD) as f:
                    img_f45 = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)
                with sftp.open(camera_image_f68_path_OLD) as f:
                    img_f68 = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)                
                
    return img_f45, img_f68

def df_color(val):
    if val is not None:
        color = 'palegreen' if 'pass' in val else 'tomato'
    else:
        color = 'lemonchiffon'
    return f'background-color: {color}'

#------------------------------------

radio_options = ['IPQC hourly report & image check', 'IPQC realtime image check', 'AA_Dev']
lottie_path = os.path.join(ROOT_DIR, 'service_cameraQualityReviewUI/lottie/*.json')
animators = glob.glob(lottie_path)
random.shuffle(animators)
ani_selected = random.choice(animators)
lottie_wait = json.load(open(ani_selected))
    
#------------------------------------
st.set_page_config(
    page_title="IPCam Quality",
    page_icon="👁",
    layout="wide",
)


with st.expander("ℹ️ about this app", expanded=False):
    st.write(
        f"""
- IPCam Quality check result display
- version: v20220824.v1
- ani_selected: {ani_selected}
"""
    )
        
st.echo()

## Range selector
st.sidebar.markdown("## 💫 **IPCam Quality Query**")
fun_mode = st.sidebar.radio('Menu', radio_options)
submit_button = st.sidebar.button(label="✨ query!")


#===== QUERAY =====
if not submit_button:
    st.stop()

#--- Partial ---
# read db result
if fun_mode==radio_options[0]:
    db = Database_Connection()
    sql_r = "select * from inference_schedule_setting"
    df = pd.read_sql(sql_r, db.engine)
    st.header("IPCam Test Log")
    c1, c2 = st.columns([20,1])
    with c1:
        st.dataframe(df.style.applymap(df_color, subset=['quality_check_result']))
    #with c2:
    #    st.image('asset/action.png')

#--- ALL ---
#reflash image
nohup_output_path = os.path.join(ROOT_DIR, 'service_cameraQualityReviewUI/tmp/nohup.out')
if fun_mode==radio_options[1]:
    if os.path.exists(nohup_output_path):
        os.remove(nohup_output_path)
    with st_lottie_spinner(lottie_wait, height=500, width=500):
        stinfo = st.info('image capture, wait about 1 minutes...')
        sh_path = os.path.join(ROOT_DIR, 'sh/f45_refresh_image.sh')
        cmd = f'nohup sh {sh_path} > {nohup_output_path} &'
        cmd_ret = os.system(cmd)
        while(True):
            cmd = f'tail -30 {nohup_output_path}'
            nohup = os.popen(cmd, 'r').read()
            field2 = st.code(nohup, language='shell')                    
            if 'camera check complete' in nohup:
                field2.empty()
                break
            time.sleep(15)
            field2.empty()
        stinfo.success('camera check complete')

#--- ALL ---
#quality check
if fun_mode==radio_options[2]:
    if os.path.exists(nohup_output_path):
        os.remove(nohup_output_path)
    #with st.spinner('Wait for AI inference, wait about 15 minutes...'):
    with st_lottie_spinner(lottie_wait, height=500, width=500):
        stinfo = st.info('AI inference, wait about 15 minutes...')
        sh_path = os.path.join(ROOT_DIR, 'sh/f45_quality_check.sh')
        cmd = f'nohup sh {sh_path} > {nohup_output_path} &'
        cmd_ret = os.system(cmd)
        while(True):
            cmd = 'nvidia-smi'
            gpu_status = os.popen(cmd, 'r').read()
            field1 = st.code(gpu_status, language='shell')        
            cmd = f'tail -30 {nohup_output_path}'
            nohup = os.popen(cmd, 'r').read()
            field2 = st.code(nohup, language='shell')        
            if 'camera check complete' in nohup:
                field1.empty()
                field2.empty()                
                break
            time.sleep(15)
            field1.empty()
            field2.empty()
        stinfo.success('camera check complete')
        
    
    df = get_camera_status()
    st.markdown("")
    st.header("IPCam Test Log (ALL)")
    c1, c2 = st.columns([5,5])
    with c1: 
        st.dataframe(df.style.applymap(df_color, subset=['detect_status']), height=600)
    #with c2:
    #    st.image('asset/action.png')

img_f45, img_f68 = get_camera_image()
tmp_imgpath_f45 = os.path.join(ROOT_DIR, 'service_cameraQualityReviewUI/tmp/camera_image_f45.jpg')
tmp_imgpath_f68 = os.path.join(ROOT_DIR, 'service_cameraQualityReviewUI/tmp/camera_image_f68.jpg')
cv2.imwrite(tmp_imgpath_f45, img_f45)
cv2.imwrite(tmp_imgpath_f68, img_f68)
st.image(tmp_imgpath_f45, use_column_width=True)
st.image(tmp_imgpath_f68, use_column_width=True)


