import sys
import pandas as pd
import os
from yaml import safe_load
import csv
from collections import OrderedDict
import cv2
import numpy as np
from tqdm import tqdm
from subprocess import call
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=all
from data_utils import get_game_entries_, process_gaze_data, create_interim_files  # nopep8

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())


RAW_DATA_DIR = config_data['RAW_DATA_DIR']
PROC_DATA_DIR = config_data['PROC_DATA_DIR']
INTERIM_DATA_DIR = config_data['INTERIM_DATA_DIR']
VALID_ACTIONS = config_data['VALID_ACTIONS']

with open(os.path.join(RAW_DATA_DIR, 'action_enums.txt'), 'r') as f:
    ACTIONS_ENUM = f.read()

games = VALID_ACTIONS.keys()
for game in games:
    create_interim_files(game=game)
