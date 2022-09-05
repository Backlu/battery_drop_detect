#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sqlalchemy import create_engine
import datetime
from common.db import Database_Connection
import cx_Oracle
import logging
from common.log import init_logging


def fetch_mes_sn():
    db = Database_Connection()
    #get mes data from oracle db
    camera_df = pd.read_sql(f'select * from f45_camera_list_map', db.engine)
    station_names = "' or station_name='".join(camera_df['mes_station'].drop_duplicates().tolist())
    station_names = "(station_name='" + station_names + "')"
    line_names = "' or line_name='".join(camera_df['mes_line'].drop_duplicates().tolist())
    line_names = "(line_name='" + line_names + "')"
    sql = "select serial_number, line_name, mo_number, model_name, group_name, station_name, in_station_time, emp_no from sfism4.r_sn_detail_t where model_name like 'W4%' and "+ station_names +" and " + line_names + " and in_station_time >= sysdate - 70/24/60/60 and in_station_time <= sysdate - 10/24/60/60"
    try:
        input_engine = create_engine("oracle://DEBBY_TING:IPcamera@1105@172.36.1.112:1521/sfsdb2", echo=False)
        result_df = pd.read_sql(sql, input_engine).drop_duplicates()
    except Exception as e:
        print(e)
        print('[ERROR !!] mes orcal db connection err')
        logging.error(str(e))
        logging.error('[ERROR !!] mes orcal db connection err')
        return
    print(f'connect MES SN DB ok, {datetime.datetime.now()}')
    logging.info(f'connect MES SN DB ok, {datetime.datetime.now()}')

    #insert to aa db
    save_df = []
    for index, row in result_df.iterrows():
        serial_number = row['serial_number']
        in_station_time = row['in_station_time']
        line_name = row['line_name']
        station_name = row['station_name']
        df = pd.read_sql(f'select * from sn_detail_min where serial_number="{serial_number}" and in_station_time="{in_station_time}" and line_name="{line_name}" and station_name="{station_name}" limit 1', db.engine)
        if len(df)==0:
            save_df.append(index)
    #insert to db
    result_df = result_df.iloc[save_df]
    with db.engine.connect() as conn, conn.begin():
        result_df.to_sql('sn_detail_min', conn, if_exists='append', index=False)
    sn_qty = len(result_df)
    print(f'save sn qty: {sn_qty}')
    logging.info(f'save sn qty: {sn_qty}')


if __name__ == '__main__':
    init_logging('fetchMesSN')
    fetch_mes_sn()
