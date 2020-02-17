import matplotlib.pyplot as plt
import re
from subprocess import call
from tqdm import tqdm
import numpy as np
import cv2
from collections import OrderedDict
import csv
from yaml import safe_load
import os
import pandas as pd

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

INTERIM_DATA_DIR = config_data['INTERIM_DATA_DIR']


def load_gaze_data(stack=1, stack_type='', stacking_skip=0, from_ix=0, till_ix=10, game='breakout', game_run='198_RZ_3877709_Dec-03-16-56-11'):

    game = game
    game_run = game_run

    game_dir = os.path.join(INTERIM_DATA_DIR, game)
    game_run_dir = os.path.join(game_dir, game_run)
    gaze_file = os.path.join(game_run_dir, game_run+'_gaze_data.csv')

    gaze_data = pd.read_csv(gaze_file)

    game_run_frames = OrderedDict({
        int(entry.split('_')[-1].split('.png')[0]): entry
        for entry in os.listdir(game_run_dir)
        if entry.__contains__('.png')
    })
    if len(game_run_frames) != len(gaze_data.index):
        unks = set(gaze_data.index).symmetric_difference(
            game_run_frames.keys())
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
    gaze_range = [160.0, 210.0]  # w,h
    gaze_data['gaze_positions'] = gaze_data['gaze_positions'].apply(lambda gps: [
                                                                    np.divide([float(co.strip()) for co in gp.split(',')], gaze_range) for gp in gps[2:-2].split('], [')])

    frame_to_gaze = gaze_data[[gaze_data.columns[1], gaze_data.columns[-1]]]

    data_ix_f = from_ix
    data_ix_t = till_ix
    images = []
    gazes = list(gaze_data['gaze_positions'])[data_ix_f:data_ix_t]

    if stack > 1:
        for frame_id in gaze_data['frame_id'][data_ix_f:data_ix_t]:
            img_data = cv2.imread(os.path.join(
                game_run_dir, frame_id+'.png'))
            images.append(img_data)
        images_ = []
        gazes_ = []
        for ix in range(len(images)-stack):
            images_.append(
                images[ix:ix+stack]
            )
            gazes_.append(
                gazes[ix:ix+stack]
            )

        return images_, gazes_

    for frame_id in gaze_data['frame_id'][data_ix_f:data_ix_t]:
        img_data = cv2.imread(os.path.join(
            game_run_dir, frame_id+'.png'))
        images.append(img_data)
    assert len(gazes) == len(images)
    return images, gazes


if __name__ == "__main__":
    load_gaze_data()
