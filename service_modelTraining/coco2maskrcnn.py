#!/usr/bin/env python
# coding: utf-8

import os
import sys
from math import ceil, floor
import numpy as np
import cv2
import pandas as pd
import json
import glob
from pathlib import Path
import logging
from common.log import init_logging
from common.utils import get_data_dir
DATA_DIR = get_data_dir()

def coco2maskrcnn(transform_folder='0_not_train_labels'):
    '''
    - Step 3.1: 將標註的 coco 格式 擷取出 battery_f 的區塊，再把 battery, vpen, black 做座標轉換成 MaskRCNN 可使用的格式
    - Step 3.1.a: 以 battery_f bbox 的 x,y 為原點(0,0)
    - Step 3.1.b: 以 battery_f bbox 的 w,h 為比例，縮放至 64x64 的大小    
    '''
    LABEL_DIR = os.path.join(DATA_DIR, 'f45_label_MaskRCNN')
    IMG_DIR = os.path.join(DATA_DIR, 'f45_output')
    label_folder = os.path.join(DATA_DIR, transform_folder)
    print(f'label transform folder:{label_folder}')
    logging.info(f'label transform folder:{label_folder}')
    
    json_paths = os.path.join(label_folder, '*.json')
    coco_labels = glob.glob(json_paths)
    debug_log_list =[]
    for label_path in coco_labels:
        with open(label_path, newline='') as f:
            data = json.load(f)

        file_path_dict = pd.DataFrame(data['images']).set_index('id')['file_name'].to_dict()
        category_type_dict = pd.DataFrame(data['categories']).set_index('id')['name'].to_dict()

        annotation_df = pd.DataFrame(data['annotations'])
        annotation_df['category_type']=annotation_df['category_id'].map(lambda x:category_type_dict[x])
        annotation_df['file_path'] = annotation_df['image_id'].map(lambda x: file_path_dict[x])

        for r in annotation_df.itertuples():
            if r.category_type!='battery_f':
                continue
            new_folder = '_'.join(Path(label_path).stem.split('_')[-2:])
            new_folder_path = os.path.join(LABEL_DIR, new_folder)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            img_file_name = os.path.basename(r.file_path)
            img_path = os.path.join(IMG_DIR, r.file_path)
            image = cv2.imread(img_path)
            x,y,w,h = r.bbox            
            battery_img = image[floor(y):ceil(y+h),floor(x):ceil(x+w)]
            battery_img_resize = cv2.resize(battery_img.copy(), (64,64))
            image[:64,:64] = battery_img_resize

            new_annotation_list =[]
            new_annotation_df = annotation_df[(annotation_df['image_id']==r.image_id) & (annotation_df['category_type'].isin(['black','battery','vpen']))]
            for r2 in new_annotation_df.itertuples():
                seg_points = r2.segmentation[0]
                seg_points = np.array([[x,y] for x,y in zip(seg_points[::2], seg_points[1::2])], np.int32)
                seg_points = seg_points.tolist()
                seg_points = [[a-floor(x), b-floor(y)] for a,b in seg_points]
                seg_points = [[a/w*64, b/h*64] for a,b in seg_points]            
                new_annotation_list.append({'label':r2.category_type,'points':seg_points})

            json_file_name = os.path.basename(img_file_name).replace('.jpg', '.json')
            json_path = os.path.join(new_folder_path, json_file_name)
            new_annotation_dict={'shapes':new_annotation_list}
            with open(json_path, 'w') as f:
                json.dump(new_annotation_dict, f, indent=2)

            new_img_path = os.path.join(new_folder_path, img_file_name)
            cv2.imwrite(new_img_path, image)   
            debug_log_list.append((new_folder, img_file_name, json_file_name))
    debug_log_df = pd.DataFrame(debug_log_list, columns=['folder','image','json'])
    print(f'label transformed qty:{len(debug_log_df)}')
    logging.info(f'label transformed qty:{len(debug_log_df)}')
    
    data_fd = ', '.join(debug_log_df['folder'].unique())    
    print(f'label folder: {data_fd}')
    logging.info(f'label folder: {data_fd}')
    
    
