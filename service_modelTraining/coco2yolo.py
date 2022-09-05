#!/usr/bin/env python
# coding: utf-8
import os, sys, glob, json
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from common.log import init_logging
from common.utils import get_data_dir
DATA_DIR = get_data_dir()
LABEL_DIR = os.path.join(DATA_DIR, 'f45_label_YOLO')

label_dict = {
    'battery_p': 0,
    'battery_f': 1,
    'battery': 2,
    'vpen': 3,
    'wz': 4,
    'ppen':5,
}

def bbox2yolov4(x,y,w,h,imgshape):
    x_center, y_center = round((x + w/2)/imgshape[1],6), round((y + h/2)/imgshape[0],6)
    width, height = round(w/imgshape[1],6), round(h/imgshape[0],6)
    return x_center, y_center, width, height


def coco2yolo(transform_folder='0_not_train_labels', dest_folder='YoloDataV2'):
    new_folder_path = os.path.join(LABEL_DIR, dest_folder)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    IMG_DIR = os.path.join(DATA_DIR, 'f45_output')
    label_folder = os.path.join(DATA_DIR, transform_folder)
    print(f'label transform folder: {label_folder}')
    logging.info(f'label transform folder: {label_folder}')

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

        for image_id in annotation_df.image_id.unique():
            img_annotation_df = annotation_df[(annotation_df['image_id']==image_id)]
            label_list = []
            for r in img_annotation_df.itertuples():
                if r.category_type not in label_dict.keys():
                    continue        
                x, y, w, h = r.bbox
                x_center, y_center, width, height = bbox2yolov4(x, y, w, h, (608, 720))
                label_list.append([label_dict[r.category_type], x_center, y_center, width, height])
            if len(label_list)==0:
                continue

            file_path = file_path_dict[image_id]
            img_file_name = os.path.basename(file_path)
            img_path = os.path.join(IMG_DIR, file_path)
            image= cv2.imread(img_path)[:608,:720]
            new_img_path = os.path.join(new_folder_path, img_file_name)
            cv2.imwrite(new_img_path, image)

            txt_file_name = img_file_name.replace('.jpg','.txt')
            new_txt_path = os.path.join(new_folder_path, txt_file_name)
            with open(new_txt_path, 'w') as f:
                for bbox in label_list:
                    bbox = [str(p) for p in bbox]
                    text = ' '.join(bbox)
                    f.write(text+'\n')
                
            debug_log_list.append((dest_folder, img_file_name, txt_file_name))
            
    debug_log_df = pd.DataFrame(debug_log_list, columns=['folder','image','json'])                    
    print(f'label transformed qty:{len(debug_log_df)}')
    logging.info(f'label transformed qty:{len(debug_log_df)}')
    return debug_log_df
    
def generate_yolo_data_txt(darknet_proj_path):
    print(f'label folder:{LABEL_DIR}')
    logging.info(f'label folder:{LABEL_DIR}')
    
    label_files = glob.glob(os.path.join(LABEL_DIR, '*/*.txt'))
    label_files = [f for f in label_files if 'train' not in f]
    img_files = list(map(lambda p: p.replace('.txt','.jpg'), label_files))
    img_files = list(filter(lambda p: os.path.exists(p), img_files))    
    df = pd.DataFrame(img_files)
    tr_df, ts_df = train_test_split(df, test_size=0.15)
    print(len(df), len(tr_df), len(ts_df))    
    darknet_data_fd = os.path.join(darknet_proj_path,'data')
    if not os.path.exists(darknet_data_fd):
        os.makedirs(darknet_data_fd)        
    yolo_data_tr = os.path.join(darknet_data_fd, 'f45_train.txt')
    yolo_data_val = os.path.join(darknet_data_fd, 'f45_test.txt')
    
    tr_df.to_csv(yolo_data_tr, header=None, index=None)
    ts_df.to_csv(yolo_data_val, header=None, index=None)
    print(f'save yolo tr data txt: {yolo_data_tr}, qty:{len(tr_df)}')
    print(f'save yolo val data txt: {yolo_data_val}, qty:{len(ts_df)}')
    logging.info(f'save yolo tr data txt: {yolo_data_tr}')
    logging.info(f'save yolo val data txt: {yolo_data_val}')    
