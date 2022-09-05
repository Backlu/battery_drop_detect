#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('.')
sys.path.append('..')
from service_AIReport.f45_hourly_check import sync_AI_detection_and_MES_sn
from service_AIReport.f45_get_mes_data_min import fetch_mes_sn
from common.log import init_logging

def test_sncy_mes_sn():
    init_logging('unit_test')
    sync_AI_detection_and_MES_sn(False)
    
def test_fetch_mesn_sn():
    init_logging('unit_test')
    fetch_mes_sn()
    