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
import h5py
from src.data.data_loaders import load_action_data, load_gaze_data
from src.features.feat_utils import transform_images, fuse_gazes_noop, reduce_gaze_stack, fuse_gazes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch

# pylint: disable=all
from data_utils import get_game_entries_, process_gaze_data  # nopep8

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())

RAW_DATA_DIR = config['RAW_DATA_DIR']
PROC_DATA_DIR = config['PROC_DATA_DIR']
INTERIM_DATA_DIR = config['INTERIM_DATA_DIR']
VALID_ACTIONS = config['VALID_ACTIONS']
STACK_SIZE = config['STACK_SIZE']
CMP_FMT = config['CMP_FMT']
OVERWRITE_INTERIM_GAZE = config['OVERWRITE_INTERIM_GAZE']

with open(os.path.join(RAW_DATA_DIR, 'action_enums.txt'), 'r') as f:
    ACTIONS_ENUM = f.read()

games = VALID_ACTIONS.keys()


def create_interim_files(game='breakout'):
    """ Reads the directories in src/data/raw directory and creates interfim files in src/data/interim directory
    
        Args:
        ----
            game -- name of the game 
        Returns:
        ----
            None     
    """
    valid_actions = VALID_ACTIONS[game]
    game_runs, game_runs_dirs, game_runs_gazes = get_game_entries_(
        os.path.join(RAW_DATA_DIR, game))

    interim_game_dir = os.path.join(INTERIM_DATA_DIR, game)
    if not os.path.exists(interim_game_dir):
        os.makedirs(interim_game_dir)

    for game_run, game_run_dir, game_run_gaze in tqdm(
            zip(game_runs, game_runs_dirs, game_runs_gazes)):
        untar_sting = 'tar -xjf {} -C {}'.format(
            os.path.join(game_run_dir, game_run) + CMP_FMT,
            interim_game_dir + '/')
        untar_args = untar_sting.split(' ')
        interim_writ_dir = os.path.join(interim_game_dir, game_run)
        gaze_out_file = '{}/{}_gaze_data.csv'.format(interim_writ_dir,
                                                     game_run)

        if os.path.exists(os.path.join(interim_game_dir, game_run)):
            print("Exists, Skipping {}/{}".format(game_run_dir, game_run))
        else:
            print("Extracting {}/{}".format(game_run_dir, game_run))
            call(untar_args)

        if not os.path.exists(gaze_out_file) or OVERWRITE_INTERIM_GAZE:
            print("Prepping gaze data for {}/{}".format(
                game_run_dir, game_run))
            gaze_file = os.path.join(game_run_dir, game_run_gaze)
            process_gaze_data(gaze_file, gaze_out_file, valid_actions)
        else:
            print("Exists, Skipping prepping of {}/{}".format(
                game_run_dir, game_run))


def create_processed_data(stack=1,
                          stack_type='',
                          stacking_skip=1,
                          from_ix=0,
                          till_ix=-1,
                          game='breakout',
                          data_types=['frames', 'actions', 'gazes']):
    """Loads data from all the game runs in the src/data/interim  directory, and creates a hdf file in the src/data/processed directory.

    
    Args:
    ----
    data_types -- types of data to save, contains atleast on of the following
                ['frames', 'actions', 'gazes', 'fused_gazes',' gazes_fused_noop']

    stack -- number of frames in the stack

    stacking_skip -- Number of frames to skip while stacking
    
    from_ix --  starting index in the data, default is first, 0
    
    till_ix -- last index of the the data to be considered, default is last ,-1

    game : game to load the data from, directory of game runs
 
    Returns:
    ----
    None
    """

    game_dir = os.path.join(INTERIM_DATA_DIR, game)
    game_runs = os.listdir(game_dir)
    images = []
    actions = []
    gaze_out_h5_file = os.path.join(PROC_DATA_DIR, game + '.hdf5')
    gaze_h5_file = h5py.File(gaze_out_h5_file, 'w')

    for game_run in tqdm(game_runs):
        print("Creating processed data for ", game, game_run)
        group = gaze_h5_file.create_group(game_run)

        images_, actions_ = load_action_data(stack, stack_type, stacking_skip,
                                             from_ix, till_ix, game, game_run)
        
        _, gazes = load_gaze_data(stack,
                                  stack_type,
                                  stacking_skip,
                                  from_ix,
                                  till_ix,
                                  game,
                                  game_run,
                                  skip_images=True)
        images_ = transform_images(images_, type='torch')

        # gazes_fused_noop = fuse_gazes_noop(images_,
        #                                    gazes,
        #                                    actions_,
        #                                    gaze_count=1,
        #                                    fuse_type='stack',
        #                                    fuse_val=0)
        gazes = torch.stack(
            [reduce_gaze_stack(gaze_stack) for gaze_stack in gazes])

        images_ = images_.numpy()
        gazes = gazes.numpy()
        gazes_fused_noop = gazes_fused_noop.numpy()

        group.create_dataset('images',
                             data=images_,
                             compression=config['HDF_CMP_TYPE'],
                             compression_opts=config['HDF_CMP_LEVEL'])
        group.create_dataset('actions',
                             data=actions_,
                             compression=config['HDF_CMP_TYPE'],
                             compression_opts=config['HDF_CMP_LEVEL'])
        group.create_dataset('gazes',
                             data=gazes,
                             compression=config['HDF_CMP_TYPE'],
                             compression_opts=config['HDF_CMP_LEVEL'])

        # group.create_dataset('gazes_fused_noop',
        #                      data=gazes_fused_noop,
        #                      compression=config['HDF_CMP_TYPE'],
        #                      compression_opts=config['HDF_CMP_LEVEL'])
        
        del gazes, images_, actions_#, gazes_fused_noop

    gaze_h5_file.close()


def combine_processed_data(game):
    """Reads the specified hdf5 file, and combines all the groups into a single combined group in the same file.

    
    Args:
    ----
    game -- name of the hdf5 file to combine, assumed to be in processed directory, without the extension
    
    Returns:
    ----
    None

    """

    gaze_out_h5_file = os.path.join(PROC_DATA_DIR, game + '.hdf5')
    gaze_h5_file = h5py.File(gaze_out_h5_file, 'a')

    groups = list(gaze_h5_file.keys())
    if not 'combined' in groups:
        all_group = gaze_h5_file.create_group('combined')
    all_group = gaze_h5_file['combined']
    data = list(gaze_h5_file[groups[0]].keys())

    for datum in tqdm(data):
        max_shape_datum = (sum([
            gaze_h5_file[group][datum].shape[0] for group in groups
            if group != 'combined'
        ]), *gaze_h5_file[groups[0]][datum].shape[1:])
        print(max_shape_datum, datum)
        all_group.create_dataset(
            datum,
            data=gaze_h5_file[groups[0]][datum][:],
            maxshape=max_shape_datum,
            compression=config['HDF_CMP_TYPE'],
        )

        for group in tqdm(groups[1:]):
            gaze_h5_file['combined'][datum].resize(
                gaze_h5_file['combined'][datum].shape[0] +
                gaze_h5_file[group][datum].shape[0],
                axis=0)
            gaze_h5_file['combined'][datum][
                gaze_h5_file['combined'][datum].
                shape[0]:, :] = gaze_h5_file[group][datum]

    gaze_h5_file.close()


if __name__ == "__main__":
    import time
    for game in games:
        create_interim_files(game=game)
        create_processed_data(stack=STACK_SIZE,
                              game=game,
                              till_ix=-1,
                              stacking_skip=1,
                              data_types=[
                                  'frames', 'actions', 'gazes', 'fused_gazes',
                                  'gazes_fused_noop'
                              ])
        # combine_processed_data(game, data_type='gazed_images')
