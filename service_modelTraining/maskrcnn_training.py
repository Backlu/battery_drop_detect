#!/usr/bin/env python
# coding: utf-8

import os
import logging
import pandas as pd
import numpy as np
import json
import glob
import cv2
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lib.mrcnn.config import Config
from lib.mrcnn import utils
from common.utils import get_data_dir, get_model_dir, gpu_ram_growth_config
import logging
from common.log import init_logging
from common.utils import Bunch
import lib.mrcnn.model as modellib

#----------
MODEL_DIR = get_model_dir()
DATA_DIR = get_data_dir()
LABEL_DIR = os.path.join(DATA_DIR, 'f45_label_MaskRCNN')


def get_tr_data():
    print(f'label folder:{LABEL_DIR}')
    logging.info(f'label folder:{LABEL_DIR}')
    
    clip_point = lambda points: [[float(np.clip(p[0],0,64)), float(np.clip(p[1],0,64))] for p in points]
    
    labeldata = glob.glob(os.path.join(LABEL_DIR, '*/*.json'))
    labeldata = [f for f in labeldata if 'train' not in f]
    df_list = []
    tr_data = []
    for p in labeldata:
        with open(p, newline='') as f:
            annotation = json.load(f)
            df = pd.DataFrame(annotation['shapes'])
            df['points'] = df['points'].map(lambda x: clip_point(x))
            annotation_clip = {'shapes':df.to_dict('records')}
            df_list.append(df)
            tr_data.append((p,annotation_clip))
    label_df = pd.concat(df_list)
    print(f'label qty:{len(label_df)}')
    logging.info(f'label qty:{len(label_df)}')    
    print(f'img qty:{len(tr_data)}')
    logging.info(f'image qty:{len(tr_data)}')        
    return label_df, tr_data
    
def get_dataset():
    label_df, data_xy = get_tr_data()
    display(label_df['label'].value_counts().to_frame())    
    data_x, data_y = zip(*data_xy)
    tr_x, val_x, tr_y, val_y = train_test_split(data_x, data_y, test_size=0.15, random_state=887 )
    tr_path = os.path.join(DATA_DIR, 'mask_rcnn_train_tmp/tr_data')
    val_path = os.path.join(DATA_DIR, 'mask_rcnn_train_tmp/val_data')
    data_bunch = Bunch()
    for x, y, folder, tag in zip([tr_x, val_x], [tr_y, val_y],[tr_path, val_path], ['tr','val']):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for json_path, annotate in zip(x, y):
            jpg_path = json_path.replace('json', 'jpg')
            new_jpg_path = os.path.join(folder, os.path.basename(jpg_path))
            new_json_path = os.path.join(folder, os.path.basename(json_path))
            image= cv2.imread(jpg_path)[:64,:64] 
            cv2.imwrite(new_jpg_path, image)
            with open(new_json_path, 'w') as f:
                json.dump(annotate, f, indent=2)
                
        dataset = BatteryDataset()
        dataset.load_dataset(tr_path)
        dataset.prepare()
        data_bunch[tag]=dataset        
        print(f'{tag}: {len(dataset.image_ids)}')

    return data_bunch
                
def get_maskrcnn_model(init_weight='coco'):
    config = BatteryConfig()
    config.LEARNING_RATE=1e-4
    #config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    init_weight = "coco" 
    if init_weight == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_weight == "coco":
        coco_model_path = os.path.join(MODEL_DIR, 'mask_rcnn_coco.h5')
        model.load_weights(coco_model_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    elif init_weight == "last":
        model.load_weights(model.find_last(), by_name=True)
    return model

def save_best_model(model):
    keras_model = model.keras_model
    keras_model.load_weights(model.checkpoint_path)
    today_str = datetime.datetime.today().strftime('%Y%m%d')
    model_path_final = os.path.join(MODEL_DIR, f'maskrcnn_{today_str}_best.h5')
    keras_model.save_weights(model_path_final)

def save_training_metric(model):
    history_dict = model.history.history
    metric_list = ['loss','rpn_class_loss','rpn_bbox_loss','mrcnn_class_loss','mrcnn_bbox_loss','mrcnn_mask_loss']
    fig = plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metric_list, start=1):
        val_metric = f'val_{metric}'
        plt.subplot(3, 2, i)
        loss = history_dict[metric]
        val_loss = history_dict[val_metric]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'g', label=metric)
        plt.plot(epochs, loss, 'g*')
        plt.plot(epochs, val_loss, 'b', label=val_metric)
        plt.plot(epochs, val_loss, 'bx')
        plt.title(metric)
        plt.ylabel(metric)
        plt.legend()
    plt.tight_layout()
    metric_jpg_path = os.path.join(model.log_dir, f'metric.jpg')
    plt.savefig(metric_jpg_path)
    plt.show()


class BatteryConfig(Config):
    # Give the configuration a recognizable name
    NAME = "battery"
    NUM_CLASSES = 1 + 3
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 64
    IMAGE_MAX_DIM = 64
    USE_MINI_MASK = False
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    TRAIN_ROIS_PER_IMAGE = 30

class InferenceConfig(BatteryConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False    
        
class BatteryDataset(utils.Dataset):
    def load_dataset(self, dataset_dir):
        self.add_class('dataset', 1, 'battery')
        self.add_class('dataset', 2, 'vpen')
        self.add_class('dataset', 3, 'black')
        
        # find all images
        j = 0
        for i, filename in enumerate(os.listdir(dataset_dir)):
            if '.jpg' in filename:
                self.add_image('dataset', 
                               image_id=j, 
                               path=os.path.join(dataset_dir, filename), 
                               annotation=os.path.join(dataset_dir, filename.replace('.jpg', '.json')))
                j += 1
                #print(j,os.path.join(dataset_dir, filename))
    
    def extract_masks(self, filename):
        json_file = os.path.join(filename)
        with open(json_file) as f:
            img_anns = json.load(f)
            
        masks = np.zeros([64, 64, len(img_anns['shapes'])], dtype='uint8')
        classes = []
        for i, anno in enumerate(img_anns['shapes']):
            mask = np.zeros([64, 64], dtype=np.uint8)
            cv2.fillPoly(mask, np.array([anno['points']], dtype=np.int32), 1)
            masks[:, :, i] = mask
            classes.append(self.class_names.index(anno['label']))
        return masks, classes
 
    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        masks, classes = self.extract_masks(path)
        return masks, np.asarray(classes, dtype='int32')
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
