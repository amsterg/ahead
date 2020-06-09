import os
import csv
from subprocess import call
from tqdm import tqdm
import pandas as pd
from yaml import safe_load
import h5py
from src.features.feat_utils import transform_images, fuse_gazes_noop
from collections import Counter
import torch
import torchvision

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

RAW_DATA_DIR = config_data['RAW_DATA_DIR']
PROC_DATA_DIR = config_data['PROC_DATA_DIR']
INTERIM_DATA_DIR = config_data['INTERIM_DATA_DIR']
CMP_FMT = config_data['CMP_FMT']
OVERWRITE_INTERIM_GAZE = config_data['OVERWRITE_INTERIM_GAZE']
VALID_ACTIONS = config_data['VALID_ACTIONS']


def get_game_entries_(game_dir):
    game_dir_entries = os.listdir(game_dir)
    game_runs = []
    game_runs_dirs = []
    game_runs_gaze = []
    for entry in game_dir_entries:
        if os.path.isdir(os.path.join(game_dir, entry)):
            rs, ds, gzs = get_game_entries_(os.path.join(game_dir, entry))
            game_runs += rs
            game_runs_dirs += ds
            game_runs_gaze += gzs
        elif entry.__contains__('.txt'):
            game_runs.append(entry.split('.txt')[0])
            game_runs_dirs.append(game_dir)
            game_runs_gaze.append(entry)
        elif entry.__contains__('.csv'):
            game_runs.append(entry.split('_gaze_data.csv')[0])
            game_runs_dirs.append(game_dir)
            game_runs_gaze.append(entry)

    return game_runs, game_runs_dirs, game_runs_gaze


def process_gaze_data(gaze_file, gaze_out_file, valid_actions):
    game_run_data = []
    with open(gaze_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            game_run_data.append(row)

    # print(game_run_data)
    header = game_run_data[0]
    game_run_data = game_run_data[1:]
    game_run_data_mod = []

    for tstep in game_run_data:
        tstep_ = []
        tstep_ = tstep[:len(header) - 1]

        # if 'null' in tstep_:
        #     if game_run_data_mod:
        #         tstep_[1:] = game_run_data_mod[-1][1:len(header) - 1]

        #         assert len(tstep_) == len(header) - 1, print(tstep_, header,
        #                                                      len(tstep_), len(header))

        gaze_data = tstep[len(header) - 1:]
        if len(gaze_data) == 1 and gaze_data[0] == 'null':

            gaze_data = game_run_data_mod[-1][len(header) - 1]
            gaze_data_ = gaze_data
            assert int(len(gaze_data) / len(gaze_data_)) == 1.0, print(
                len(gaze_data), len(gaze_data_))
        else:
            gaze_data_ = [[float(gd) for gd in gaze_data[ix:ix + 2]]
                          for ix in range(0,
                                          len(gaze_data) - 1, 2)]
            assert int(len(gaze_data) / len(gaze_data_)) == 2.0, print(
                len(gaze_data), len(gaze_data_))
        tstep_.append(gaze_data_)
        assert len(tstep_) == len(header)
        game_run_data_mod.append(tstep_)

    game_run_data_mod_df = pd.DataFrame(game_run_data_mod, columns=header)
    game_run_data_mod_df['action'] = game_run_data_mod_df['action'].apply(
        lambda x: 0 if x == 'null' else (x if int(x) in valid_actions else 0))

    # frame_ids = game_run_data_mod_df['frame_id']
    # assert len(frame_ids) == len(game_run_frames), print(len(frame_ids),
    #                                                     len(game_run_frames))

    game_run_data_mod_df.to_csv(gaze_out_file)


def stack_data(images, targets, stack=1, stack_type='', stacking_skip=1):
    if images:
        assert len(images) == len(targets)
    if stack > 0:
        images_ = []
        targets_ = []
        for ix in range(0,len(targets) - stack,stacking_skip):
            images_.append(images[ix:ix + stack])
            targets_.append(targets[ix:ix + stack])
        return images_, targets_
        
    if images:
        assert len(images) == len(targets)
    return images, targets



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)
        label_to_count = Counter(dataset.labels)

        weights = [1.0 / label_to_count[ix] for ix in dataset.labels]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples