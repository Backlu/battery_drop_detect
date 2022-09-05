#!/usr/bin/env python
# coding: utf-8

import sys, os
sys.path.append('.')
from common.utils import get_data_dir, gpu_ram_config, clean_folder
from common.log import init_logging
from service_cameraQualityCheck.cam_quality_check_main import f45_cam_check, f68_cam_check
from service_cameraQualityCheck.cam_image_reflash_main import reflash_f45_img, reflash_f68_img
DATA_DIR = get_data_dir()


def test_f45_cam_quality_check():
    init_logging('unit_test')
    output_dir = os.path.join(DATA_DIR, 'camera_quality_check')
    clean_folder(output_dir)
    f45_cam_check(debug=True)
    clean_folder(output_dir)
    
def test_f68_cam_quality_check():
    init_logging('unit_test')
    output_dir = os.path.join(DATA_DIR, 'camera_quality_check')
    clean_folder(output_dir)
    f68_cam_check(debug=True)
    clean_folder(output_dir)
    
def test_f45_cam_image_reflash():
    init_logging('unit_test')
    output_dir = os.path.join(DATA_DIR, 'camera_quality_check')
    clean_folder(output_dir)    
    reflash_f45_img(debug=True)
    clean_folder(output_dir)
    
def test_f68_cam_image_reflash():
    init_logging('unit_test')
    output_dir = os.path.join(DATA_DIR, 'camera_quality_check')
    clean_folder(output_dir)    
    reflash_f68_img(debug=True)  
    clean_folder(output_dir)
    
    