import pandas as pd
import os
from yaml import safe_load
import csv
from collections import OrderedDict
import cv2
import numpy as np
from tqdm import tqdm
from subprocess import call
import re

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

raw_data_dir = config_data['raw_data_dir']
proc_data_dir = config_data['proc_data_dir']
interim_data_dir = config_data['interim_data_dir']

with open(os.path.join(raw_data_dir, 'action_enums.txt'), 'r') as f:
    actions_enum = f.read()

# game_0 = 'breakout'
game = 'breakout'
valid_actions = config_data['valid_actions'][game]
print(game, valid_actions)

game_run = '58_RZ_2489381_Aug-11-17-37-10'
game_dir = os.path.join(interim_data_dir, game)
game_run_dir = os.path.join(game_dir, game_run)
gaze_file = os.path.join(game_run_dir, game_run+'_gaze_data.csv')

gaze_data = pd.read_csv(gaze_file)

# print(gaze_data)

game_run_frames = OrderedDict({
    int(entry.split('_')[-1].split('.png')[0]): entry
    for entry in os.listdir(game_run_dir)
    if entry.__contains__('.png')
})
if len(game_run_frames) != len(gaze_data.index):
    unks = set(gaze_data.index).symmetric_difference(game_run_frames.keys())
    unks_ = []
    for unk in unks:
        if unk in game_run_frames:
            del game_run_frames[unk]
        else:
            unks_.append(unk)

    gaze_data = gaze_data.drop([gaze_data.index[unk] for unk in unks_])
    assert len(game_run_frames) == len(gaze_data.index), print(
        len(game_run_frames), len(gaze_data.index))
    assert set(game_run_frames.keys()) == set(gaze_data.index)
gaze_data['gaze_positions'] = gaze_data['gaze_positions'].apply(lambda gps: [
                                                                [float(co.strip()) for co in gp.split(',')] for gp in gps[2:-2].split('], [')])

# print(type(gaze_data['gaze_positions'].iloc[0]))
# print(gaze_data['gaze_positions'].iloc[0])

frame_to_gaze = gaze_data[[gaze_data.columns[1],gaze_data.columns[-1]]]
print(frame_to_gaze)

