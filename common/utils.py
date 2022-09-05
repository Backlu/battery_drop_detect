#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import tensorflow as tf
import glob

def get_project_root():
    return Path(__file__).parent.parent

def get_config_dir():
    root = get_project_root()
    config_path = os.path.join(root,'config','config.ini')
    return config_path

def get_data_dir():
    return '/mnt/hdd1/Data/f45movement'

def get_model_dir():
    return '/mnt/hdd1/Model/f45movement/model'

def gpu_ram_config(gpu_id=None, ram=None):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    assert len(gpus)>0, 'No GPU can be config'
    
    gpu_id = int(os.environ['ALIYUN_COM_GPU_MEM_IDX']) if gpu_id is None else gpu_id
    memory_limit = int(os.environ['ALIYUN_POD_GPU_MEMORY']) 
    preserve_memory = int(os.environ['PRESERVE_GPU_MEMORY']) if ram is None else ram
    print(f'gpu_id:{gpu_id}')
    print(f'memory_limit:{memory_limit}')
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[gpu_id],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=preserve_memory)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        
def gpu_ram_growth_config(gpu_id=None):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    gpu_id = int(os.environ['ALIYUN_COM_GPU_MEM_IDX']) if gpu_id is None else gpu_id
    print(f'gpu_id:{gpu_id}')
    try:
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        
def clean_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        files = glob.glob(f'{path}/*')
        for file in files:
            os.remove(file)        

class Bunch(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass            