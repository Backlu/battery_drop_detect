#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
import cv2

def f45_camera_imgplot(mid_list, img_list, save_path, debug):
    plt.figure(figsize=(18,48))
    for i, (mid, img) in enumerate(zip(mid_list, img_list), start=1):
        cv2.rectangle(img, (510, 280), (550, 320), (221,160,221), 6)
        cv2.line(img, (507, 277), (1280, 277), (255, 255, 0), 5)
        cv2.line(img, (507, 277), (507, 720), (255, 255, 0), 5)
        cv2.line(img, (350, 175), (1280, 175), (255, 255, 0), 5)
        cv2.line(img, (350, 175), (350, 720), (255, 255, 0), 5)    
        plt.subplot(12,3,i)
        plt.axis('off') 
        plt.imshow(img)
        plt.title(f'{mid}')
    plt.rcParams["savefig.jpeg_quality"] = 75
    plt.savefig(save_path, bbox_inches='tight')
    if debug==False:
        os.system(f"scp {save_path} elf@10.142.3.58:/mnt/hdd1/ipqc")
    
def f68_camera_imgplot(mid_list, img_list, save_path, debug):
    plt.figure(figsize=(18,48))
    for i, (mid, img) in enumerate(zip(mid_list, img_list), start=1):
        cv2.line(img, (550,297), (640,560), (255, 255, 0), 10)
        plt.subplot(12,3,i)
        plt.axis('off') 
        plt.imshow(img)
        plt.title(f'{mid}')
    plt.rcParams["savefig.jpeg_quality"] = 75
    plt.savefig(save_path, bbox_inches='tight')
    if debug==False:    
        os.system(f"scp {save_path} elf@10.142.3.58:/mnt/hdd1/ipqc")
        