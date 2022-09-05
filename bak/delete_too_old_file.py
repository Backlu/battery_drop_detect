#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import datetime
import time


# In[20]:

# F68: /mnt/hdd1/opmovement/output
last = datetime.datetime.now() - datetime.timedelta(days=150)
last_unix = last.timestamp()
files = glob.glob('/mnt/hdd1/opmovement/output/*/*/*')
for file in files:
    file_time = os.stat(file).st_mtime
    if file_time < last_unix:
        print(f'rm -f {file}')
        os.system(f'rm -f {file}') 

# F45: /mnt/hdd1/f45_output_test
last = datetime.datetime.now() - datetime.timedelta(days=150)
last_unix = last.timestamp()
files = glob.glob('/mnt/hdd1/f45_output_test/*/*')+glob.glob('/mnt/hdd1/f45_output_test/*/*/*')+glob.glob('/mnt/hdd1/f45_output_test/*/*/*/*')
for file in files:
for file in files:
    file_time = os.stat(file).st_mtime
    if file_time < last_unix:
        print(f'rm -f {file}')
        os.system(f'rm -f {file}') 
        
# F45: /mnt/hdd1/f45_output
last = datetime.datetime.now() - datetime.timedelta(days=150)
last_unix = last.timestamp()
files = glob.glob('/mnt/hdd1/f45_output/*/*')+glob.glob('/mnt/hdd1/f45_output/*/*/*')+glob.glob('/mnt/hdd1/f45_output/*/*/*/*')
for file in files:
for file in files:
    file_time = os.stat(file).st_mtime
    if file_time < last_unix:
        print(f'rm -f {file}')
        os.system(f'rm -f {file}') 

print('done', datetime.datetime.now())
