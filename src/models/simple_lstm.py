import pandas as pd
import os
from yaml import safe_load
import csv
from collections import OrderedDict
import cv2
import numpy as np
from tqdm import tqdm
from subprocess import call
f

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

raw_data_dir = config_data['raw_data_dir']
proc_data_dir = config_data['proc_data_dir']
print(raw_data_dir)
with open(os.path.join(raw_data_dir, 'action_enums.txt'), 'r') as f:
    actions_enum = f.read()

# game_0 = 'breakout'
game = 'breakout'
print(game)
#game_0_run_0
game_runs = [
    entry.split('.txt')[0]
    for entry in os.listdir(os.path.join(raw_data_dir, game))
    if entry.__contains__('.txt')
]

game_run = game_runs[0]
print(game_run)
game_run = '490_KM_3486399_Jul-18-16-47-55'
proc_data_dir = os.path.join(os.path.join(proc_data_dir, game),
                             game_run) + '_wgz'

processed_game_data = pd.read_csv(
    '{}/processed_game_data'.format(proc_data_dir))
print(processed_game_data)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
