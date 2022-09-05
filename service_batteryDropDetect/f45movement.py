#!/usr/bin/env python
# coding: utf-8

# f45_opmovement
# Copyright © 2021 AA

# History:
# 03/23: initial commit
# 05/03: YOLOv4tiny+MaskRCNN
# 05/04: 存兩張(原始image和inference image)
# 05/10: 用ip分別存database table(61 -> f45_anomaly_info; 60 -> f45_anomaly_info_rtsp)
# 05/11: 更新yolo: yolov4-tiny-f45-0507_best.weights
# 22/03/17: 用inference_schedule_setting表決定是否繼續inference
# 22/04/14: update yolo
# 22/06/24: refactor
# 22/08/18: refactoring, JayHsu

# Please refer to https://hackmd.io/@Yvonne/r1OYvj-qc for detail explanation

import os, cv2, time, datetime
import numpy as np
import pandas as pd
from sqlalchemy import Table, Column, String, Integer, MetaData, insert, Text
import pymysql
import threading, multiprocessing
import logging
import service_batteryDropDetect.darknet as darknet
from lib.mrcnn.config import Config
from lib.mrcnn import utils
import lib.mrcnn.model as modellib
from lib.mrcnn.model import log
from common.db import Database_Connection
from common.utils import get_model_dir
MODEL_DIR = get_model_dir()

MAIN_VER='22/04/14(Yvonne)'
YOLO_VER='yolov4-tiny-f45-0721_best.weights'
MASKRCNN_VER='maskrcnn_0505v2_best.h5'


class F45Movement(object):
    STEAM_MODE_RTSP=1
    STEAM_MODE_REPLAY=2
    STEAM_MODE_TEST_FILE=3
    STEAM_MODE_PREVIEW=4
    
    _defaults = {
        'is_online': False,
        'stream_mode': False,
        
        'yolo_configPath': os.path.join(MODEL_DIR, 'yolov4-tiny-f45-0331.cfg'),
        'yolo_weightPath': os.path.join(MODEL_DIR, 'yolov4-tiny-f45-0721_best.weights'),
        'yolo_metaPath' : os.path.join(MODEL_DIR, 'f45_obj.data'),
        'maskrcnn_modelPath': os.path.join(MODEL_DIR, 'maskrcnn_0505v2_best.h5'),
        
        # Object color of YOLO model
        'classid_dict': {
            'battery_p':(127,255,50),
            'battery_f':(147,20,255),
            'battery':(135,138,128),
            'vpen':(153, 255,255),
            'wz':(221,160,221),
            'ppen':(255,255,86)
        },
        # Mask color of MaskRCNN model - classID: [class name, color]
        'classid_mask': {
            0: ['BG',(255,255,255)],
            1: ['battery',(135,138,128)],
            2: ['vpen',(153, 255,255)],
            3: ['black',(0,0,0)]
        }
    }
    def __init__(self, camera_mid=None, dir_draw=None, dir_raw=None, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.camera_mid = camera_mid
        self.init_yolo()
        self.init_rcnn()
        self.init_db_table_name()
        self.init_detect_param()
        self.dir_draw = dir_draw
        self.dir_raw = dir_raw
        self.detect_cnt = 0
        self.detect_black_cnt = 0
    
    def init_detect_param(self):
        self.bf_record = 0
        self.bk_record = 0
        self.t_record = datetime.datetime.now()
        self.save_image_draw = None
        self.save_image_raw = None
        self.camera_quality_good= None
        
    def init_yolo(self):
        print(f'Load YOLO Model from {self.yolo_weightPath}')
        logging.info(f'Load YOLO Model from {self.yolo_weightPath}')
        self.network, self.class_names, self.class_colors = darknet.load_network(self.yolo_configPath, self.yolo_metaPath, self.yolo_weightPath, batch_size=1)
        net_w, net_h = darknet.network_width(self.network), darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(net_w, net_h, 3)
        self.net_wh = (net_w, net_h)
    
    def init_rcnn(self):
        inference_config = InferenceConfig()
        self.mask_model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                           model_dir='model')
        print(f'Load MaskRCNN Model from {self.maskrcnn_modelPath}')
        logging.info(f'Load MaskRCNN Model from {self.maskrcnn_modelPath}')
        self.mask_model.load_weights(self.maskrcnn_modelPath, by_name=True)

    def init_db_table_name(self):
        if self.is_online:
            self.table = 'f45_anomaly_info'
        else:
            self.table = 'f45_anomaly_info_rtsp'


#--------------main-------------------                
    def detect(self, fid, image, save_log=False):
        self.fid = fid
    # ----- determine if inference or not based on quality check results --------------------------------
    # 每小時的30分時去看Database內的quality check結果，如果Pass才繼續inference，否則停止inference
        self.get_camera_quality()
        if (self.camera_quality_good == False)&(self.stream_mode==self.STEAM_MODE_RTSP):
            self.bf_record = 0
            self.bk_record = 0
            self.t_record = datetime.datetime.now()
            self.save_image_draw = None
            self.save_image_raw = None
            return
    # ----------------------------------------------------------
        fstart = time.time()
        # YOLO detection
        detections = self.yolo_detect(image, self.darknet_image, self.net_wh)
        # Check YOLO detection fulfill requirements
        bat_img, f_score = self.yolo_bat(image, detections)
        if bat_img is not None:
            # MaskRCNN detection
            results = self.mask_model.detect([bat_img], verbose=0)
            black_detected, detect_ret = self.rcnn_bat(results)
            self.detect_cnt+=1            
            if not black_detected:
                return
            
            k_score = detect_ret['black'][1]
            maskrcnn_ret = results[0]
            # ----- decide when to save the image and which image to be saved -----------------------------------------
            if (datetime.datetime.now() - self.t_record).total_seconds() > 3.5: #判斷是否同一"次"
                if self.save_image_draw is not None:
                    #只存上一次fail的第一張圖片
                    multiprocessing.Process(target = cv2.imwrite, args = (self.jpg_path_draw, self.save_image_draw)).start()
                    multiprocessing.Process(target = cv2.imwrite, args = (self.jpg_path_raw, self.save_image_raw)).start()
                    multiprocessing.Process(target = self.to_database, args = (self.camera_mid, self.first_fail_time, 'battery_fail', self.jpg_path)).start()
                self.bf_record = 0
                self.bk_record = 0
                
            self.t_record = datetime.datetime.now()  
            #是否要存該張frame: 加權評估
            if ((f_score - self.bf_record)*12 + (k_score - self.bk_record)) <= 0: 
                return
            
            self.detect_black_cnt+=1
            self.bf_record = f_score
            self.bk_record = k_score
            # ----------------------------------------------------------------------
            # calculate fps
            end = time.time()
            seconds = end - fstart   
            fps = np.round(1/seconds, 2)
            
            # draw image and output image
            # FIXME: 這邊大部分都多做的?
            self.save_image_raw = image.copy()
            draw_info = {'bat_img':bat_img, 'maskrcnn_ret':maskrcnn_ret, 'detections':detections, 'detect_ret':detect_ret, 'fps':fps}
            self.save_image_draw = self.draw_img(image, draw_info)
            self.first_fail_time = datetime.datetime.now()
            now_str = self.first_fail_time.strftime('%Y-%m-%d-%H-%M-%S.%f')[:-5]    
            self.jpg_path_draw = os.path.join(self.dir_draw, f'{self.camera_mid}_{now_str}.jpg')
            self.jpg_path_raw = os.path.join(self.dir_raw, f'{self.camera_mid}_{now_str}.jpg')

    def draw_img(self, image, draw_info):
        maskrcnn_ret = draw_info['maskrcnn_ret']
        fps = draw_info['fps']
        mergeimg = self.draw_mask(draw_info['bat_img'], maskrcnn_ret['rois'], maskrcnn_ret['masks'], maskrcnn_ret['class_ids'], maskrcnn_ret['scores'])
        image[:64,:128] = mergeimg
        image = self.draw_bbox(image, draw_info['detections'])
        image = self.draw_maskscore(image, maskrcnn_ret, draw_info['detect_ret'])
        cv2.putText(image, f'FPS:{fps}', (10,530), cv2.FONT_HERSHEY_DUPLEX,0.6, (255,255,128), 1)
        return image
    
    def get_camera_quality(self):
        onhour = (datetime.datetime.now().minute==30) and (datetime.datetime.now().second==0)
        onhour_recheck = True if onhour and (self.fid%15==1) else False 
        
        if (onhour_recheck) or (self.camera_quality_good is None):
            db = Database_Connection()
            sql_r = "select * from inference_schedule_setting where mid = %(mid)s"
            df = pd.read_sql(sql_r, db.engine, params={'mid':self.camera_mid})
            if len(df) > 0:
                status = df['quality_check_result'][0]
            else:
                status = 'not in inference list'
            if status == 'camera pass':
                self.camera_quality_good = True
            else:
                self.camera_quality_good = False
            print(f'[get_camera_quality] {self.camera_mid}, {status}, {self.camera_quality_good}')
            logging.info(f'[get_camera_quality] {self.camera_mid}, {status}, {self.camera_quality_good}')
    
#--------------yolo funs-------------------                        
    def yolo_detect(self, image, darknet_image, net_wh):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, net_wh, interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=0.1, nms=.2)
        return detections

    def yolo_bat(self, image, detections):
        #detectoion資料結構: [type, score, (x,y,w,h)]
        bat_img = None
        f_score = 0
        battery_p = list(filter(lambda x: x[0]=='battery_p', detections))
        if len(battery_p) > 0:
            return bat_img, f_score
        battery_f = list(filter(lambda x: x[0]=='battery_f', detections))
        battery_f = sorted(battery_f, key = lambda x: float(x[1]), reverse=True)
        battery_f = list(filter(lambda x: float(x[1]) >= 30, battery_f)) # battery_f 的分數限制
        if len(battery_f) > 0:
            f_score = float(battery_f[0][1])
            r_w, r_h = (image.shape[1]/self.net_wh[0], image.shape[0]/self.net_wh[1])
            x, y, w, h = float(battery_f[0][2][0]), float(battery_f[0][2][1]), float(battery_f[0][2][2]), float(battery_f[0][2][3])
            x=x*r_w
            w=w*r_w
            y=y*r_h
            h=h*r_h
            x1 = int(round(x - (w / 2)))
            y1 = int(round(y - (h / 2)))
            x2 = int(round(x + (w / 2)))
            y2 = int(round(y + (h / 2))) 
            #原始的中心點
            cx, cy = (x1+x2)//2, (y1+y2)//2
            x1 = np.clip(x1, 0, image.shape[1])
            y1 = np.clip(y1, 0, image.shape[0])
            x2 = np.clip(x2, 0, image.shape[1])
            y2 = np.clip(y2, 0, image.shape[0])
            bat_img = cv2.resize(image[y1:y2, x1:x2], (64,64))    
        return bat_img, f_score        
    
                  
#--------------rcnn funs-------------------
    def rcnn_bat(self, results):
        black_detected = True
        detect_ret= {}
        r = results[0]
        a = sorted(r['class_ids'].tolist())
        if all(x in a for x in [1,2,3]) is False: # 確認(1): battery, black, vpen都有被偵測到
            black_detected=False
            return black_detected, detect_ret
        
        for i in range(r['masks'].shape[2]):
            cls_name = self.classid_mask[r['class_ids'][i]][0]
            if cls_name in detect_ret.keys():
                if detect_ret[cls_name][2] > r['scores'][i]:
                    continue            
            mask = r['masks'][:,:,i].astype(np.uint8)
            mask*=255
            cX, cY, area = self.mask_feature(mask)
            if area == 0:
                continue
            detect_ret[self.classid_mask[r['class_ids'][i]][0]]=(cX, cY), area, r['scores'][i]

        k_score = detect_ret['black'][1]
        if k_score < 400: # 確認(2): black的限制
            black_detected=False
            return black_detected, detect_ret
        
        return black_detected, detect_ret
    
    def mask_feature(self, mask):
        contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse=True)
        if len(contours) == 0:
            return 0,0,0
        area = cv2.contourArea(contours[0])
        if area == 0:
            return 0,0,0
        M = cv2.moments(contours[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY, area

    
#--------------draw-------------------                
    def draw_debuginfo(self, image, fid, cls):
        id_map = {x:self.classid_dict[x][0] for x in self.classid_dict.keys()}
        fontsize=0.7
        color=(255,255,128)
        lspace=20
        msg_w, msg_h = 5, 90
        if cls is not None:
            cv2.putText(image, f'cls:{[id_map[k] for k in cls.keys()]}', (10,130) , cv2.FONT_HERSHEY_DUPLEX , fontsize, color,1)
            cv2.putText(image, f'score:{[int(v[0]*100) for v in cls.values()]}', (10,160) , cv2.FONT_HERSHEY_DUPLEX , fontsize, color,1)
        return image     

    def draw_bbox(self, image, detections):
        for detection in detections:
            r_w, r_h = (image.shape[1]/self.net_wh[0], image.shape[0]/self.net_wh[1])
            x, y, w, h = float(detection[2][0]), float(detection[2][1]), float(detection[2][2]), float(detection[2][3])
            x=x*r_w
            w=w*r_w
            y=y*r_h
            h=h*r_h
            x1 = int(round(x - (w / 2)))
            y1 = int(round(y - (h / 2)))
            x2 = int(round(x + (w / 2)))
            y2 = int(round(y + (h / 2))) 
            #原始的中心點
            cx, cy = (x1+x2)//2, (y1+y2)//2
            x1 = np.clip(x1, 0, image.shape[1])
            y1 = np.clip(y1, 0, image.shape[0])
            x2 = np.clip(x2, 0, image.shape[1])
            y2 = np.clip(y2, 0, image.shape[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), self.classid_dict[detection[0]], 1)
            idx = detections.index(detection)
            cv2.putText(image, str(detection[:2]), (10,130+30*idx) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,128),1)
        cv2.putText(image, f'FID:{self.fid}', (10,550), cv2.FONT_HERSHEY_DUPLEX,0.7, (255,255,128), 1)
        return image

    def draw_fail(self, image, fps):
        cv2.putText(image, 'Battery Fail', (10,300), cv2.FONT_HERSHEY_DUPLEX,1.5, (0,0,255), 1)
        cv2.putText(image, f'FPS:{fps}', (10,530), cv2.FONT_HERSHEY_DUPLEX,0.7, (255,255,128), 1)
        return image

    def draw_mask(self, image, boxes, masks, class_ids, score):
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            class_id = class_ids[i]
            color = self.classid_mask[class_id][1]
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # Mask
            mask = masks[:, :, i]
            masked_image = self._apply_mask(masked_image, mask, color)
        mergeimg = np.hstack((image, masked_image))
        return mergeimg
    
    def _apply_mask(self, image, mask, color, alpha=0.5):
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] * (1 - alpha) + alpha * color[c] * 1,
                                      image[:, :, c])
        return image    
    
    def draw_maskscore(self, image, r, detect_ret):
        a = dict(zip(r['class_ids'], r['scores']))
        alist = [(self.classid_mask[key][0], int(a[key]*100)) for key in a.keys()] 
        for value in alist:
            idx = alist.index(value)
            cv2.putText(image, str(value), (10,300+30*idx) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (221,160,221),1)
        dlist = [(key, detect_ret[key]) for key in detect_ret.keys()] 
        for value in dlist:
            idx = dlist.index(value)
            cv2.putText(image, str(value[:2]), (10,400+30*idx) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (221,160,221),1)
        return image
    
#--------------postprocessing-------------------            
    
    def to_database(self, vid, error_time, error_type, jpg_link):
        db = Database_Connection()
        metadata = MetaData(bind=db.engine)
        anomaly_info = Table(self.table, metadata,
                             Column('vid', String(50), primary_key=True),   
                             Column('error_time', String(50),  primary_key=True), 
                             Column('error_type', String(50)),
                             Column('jpg_link', Text),
                            )
        metadata.create_all(db.engine)
        conn = db.engine.connect()
        act = insert(anomaly_info).values(vid=vid,
                                          error_time = error_time,
                                          error_type = error_type,
                                          jpg_link = jpg_link
                                         )
        conn.execute(act)
        conn.close()
        
# ---------- detect workzone ------
    def detect_workzone(self, image):
        detections = self.yolo_detect(image, self.darknet_image, self.net_wh)
        word_zone_list = list(filter(lambda x: x[0]=='wz', detections))
        word_zone_list = sorted(word_zone_list, key = lambda x: float(x[1]), reverse=True)
        if len(word_zone_list) > 0:
            wz_biggest = word_zone_list[0]
            wz_score = wz_biggest[1]
            wz_x, wz_y, wz_w, wz_h = wz_biggest[2]
        else:
            wz_score=-1
            wz_x=wz_y=wz_w=wz_h=-1
        return wz_score, (wz_x, wz_y)
        

class BatteryConfig(Config):
    # Give the configuration a recognizable name
    NAME = "battery"
    
    NUM_CLASSES = 1 + 3

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 64
    IMAGE_MAX_DIM = 64
    USE_MINI_MASK = False
    RPN_ANCHOR_SCALES = (2, 4, 8, 16, 32)

class InferenceConfig(BatteryConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False

        