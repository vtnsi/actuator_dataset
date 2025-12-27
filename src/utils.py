# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:46:28 2025

"""

import json


def read_configs(config_filename):
    with open(config_filename, "r") as f:
        config_data = json.load(f)
    return config_data

def save_configs(config_data, filename):
    with open(filename, "w") as f:
        json.dump(config_data, f)
