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


def load_gaze_data(stack=1):
    with open('src/config.yaml', 'r') as f:
        config_data = safe_load(f.read())

    raw_data_dir = config_data['raw_data_dir']
    proc_data_dir = config_data['proc_data_dir']
    interim_data_dir = config_data['interim_data_dir']

    with open(os.path.join(raw_data_dir, 'action_enums.txt'), 'r') as f:
        actions_enum = f.read()

    # game_0 = 'breakout'
    game = 'breakout'
    game_run = '198_RZ_3877709_Dec-03-16-56-11'

    valid_actions = config_data['valid_actions'][game]

    game_dir = os.path.join(interim_data_dir, game)
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
    data_ix_f = 0
    data_ix_t = 10
    images = []
    rand_ixs = np.random.randint(
        frame_to_gaze.shape[0], size=data_ix_t-data_ix_f)
    gazes = list(gaze_data['gaze_positions'])[data_ix_f:data_ix_t]
    gaze = gazes[0]

    if stack > 1:
        for frame_id in gaze_data['frame_id'][data_ix_f:data_ix_t]:
            img_data = cv2.imread(os.path.join(
                game_run_dir, frame_id+'.png'))
            # cv2.imshow('rand_image', img_data)
            # cv2.waitKey()
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
        # cv2.imshow('rand_image', img_data)
        # cv2.waitKey()
        images.append(img_data)
    assert len(gazes) == len(images)
    return images, gazes


if __name__ == "__main__":
    load_gaze_data()
