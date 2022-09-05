#!/usr/bin/env python
# coding: utf-8

import os
from sqlalchemy import create_engine
import configparser
from common.utils import get_config_dir
CONFIG_PATH = get_config_dir()



class Database_Connection(object):
    _defaults = {
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self._get_db_config()
        self._get_engine()

    def _get_db_config(self):
        self.db_info = {}
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        #FIXME: seperate dev and prod env (Jay, 20220818)
        config_name = 'db_tpe'
        self.db_info['DESC'] = config.get(config_name, 'desc')
        self.db_info['DB_IP'] = config.get(config_name, 'db_ip')
        self.db_info['DB_PORT'] = config.get(config_name, 'db_port')
        self.db_info['DB_USERNAME'] = config.get(config_name, 'db_username')
        self.db_info['DB_PASSWORD'] = config.get(config_name, 'db_password')
        self.db_info['DB_NAME'] = config.get(config_name, 'db_name')
        
    def _get_engine(self):
        self.engine = create_engine(f"mysql+pymysql://{self.db_info['DB_USERNAME']}:{self.db_info['DB_PASSWORD']}@{self.db_info['DB_IP']}:{self.db_info['DB_PORT']}/{self.db_info['DB_NAME']}?charset=UTF8MB4")

