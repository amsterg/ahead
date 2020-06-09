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
import h5py
from src.data.data_utils import stack_data, transform_images
from src.features.feat_utils import fuse_gazes_noop, fuse_gazes
from torch.utils import data
import torch
from src.data.data_utils import ImbalancedDatasetSampler
from itertools import cycle
from collections import Counter

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())

INTERIM_DATA_DIR = config['INTERIM_DATA_DIR']
PROC_DATA_DIR = config['PROC_DATA_DIR']


def load_pp_data(game='breakout', game_run='198_RZ_3877709_Dec-03-16-56-11'):
    """Loads interim data for the specified game and game run 
    
    Args:
    ----
     game : game to load the data from, directory of game runs
     game_run : game_run to load the data from, directory of frames and gaze data

    Returns:
    ----
     gaze_data :  gaze data for the specified game run
     game_run_dir : directory conatining game run frames
    
    """
    game = game
    game_run = game_run

    game_dir = os.path.join(INTERIM_DATA_DIR, game)
    game_run_dir = os.path.join(game_dir, game_run)
    gaze_file = os.path.join(game_run_dir, game_run + '_gaze_data.csv')

    gaze_data = pd.read_csv(gaze_file)

    game_run_frames = OrderedDict({
        int(entry.split('_')[-1].split('.png')[0]): entry
        for entry in os.listdir(game_run_dir) if entry.__contains__('.png')
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
    return gaze_data, game_run_dir


def load_gaze_data(stack=1,
                   stack_type='',
                   stacking_skip=1,
                   from_ix=0,
                   till_ix=-1,
                   game='breakout',
                   game_run='198_RZ_3877709_Dec-03-16-56-11',
                   skip_images=False):
    """Loads and processes gaze data/images for the specified game and game run
    
    Args:
    ----
     stack : Number of frames to stack
     stack_type : ',
     stacking_skip : Number of frames to skip while stacking,
     from_ix :  starting index in the data, default is first, 0
     
     till_ix : last index of the the data to be considered, default is last ,-1
     
     game : game to load the data from, directory of game runs
     game_run : game_run to load the data from, directory of frames and gaze data
 
     skip_images= if True doesn't return frames for the gaem run
 
    Returns:
    ----
     images_ : None or images for the specified game run
     gazes_ :  Formatted gaze data for the specified game run
    
    """
    gaze_data, game_run_dir = load_pp_data(game=game, game_run=game_run)
    gaze_range = [160.0, 210.0]  # w,h
    gaze_data['gaze_positions'] = gaze_data['gaze_positions'].apply(
        lambda gps: [
            np.divide([float(co.strip()) for co in gp.split(',')], gaze_range)
            for gp in gps[2:-2].split('], [')
        ])
    data_ix_f = from_ix
    data_ix_t = till_ix
    images = []
    gazes = list(gaze_data['gaze_positions'])[data_ix_f:data_ix_t]
    if not skip_images:
        for frame_id in gaze_data['frame_id'][data_ix_f:data_ix_t]:
            img_data = cv2.imread(os.path.join(game_run_dir,
                                               frame_id + '.png'))
            images.append(img_data)

    images_, gazes_ = stack_data(images,
                                 gazes,
                                 stack=stack,
                                 stack_type=stack_type,
                                 stacking_skip=stacking_skip)
    return images_, gazes_


def load_action_data(stack=1,
                     stack_type='',
                     stacking_skip=1,
                     from_ix=0,
                     till_ix=10,
                     game='breakout',
                     game_run='198_RZ_3877709_Dec-03-16-56-11'):
    """Loads and processes action data/images for the specified game and game run

    Args:
    ----
     stack : Number of frames to stack
     stack_type : ',
     stacking_skip : Number of frames to skip while stacking,
     from_ix :  starting index in the data, default is first, 0
     
     till_ix : last index of the the data to be considered, default is last ,-1
     
     game : game to load the data from, directory of game runs
     game_run : game_run to load the data from, directory of frames and gaze data
 
    Returns:
    ----
     images_ : images for the specified game run
     action_ : action data for the specified game run
    
    """
    gaze_data, game_run_dir = load_pp_data(game=game, game_run=game_run)
    data_ix_f = from_ix
    data_ix_t = till_ix
    images = []

    actions = list(gaze_data['action'])[data_ix_f:data_ix_t]
    for frame_id in gaze_data['frame_id'][data_ix_f:data_ix_t]:
        img_data = cv2.imread(os.path.join(game_run_dir, frame_id + '.png'))
        images.append(img_data)

    images_, actions_ = stack_data(images,
                                   actions,
                                   stack=stack,
                                   stack_type=stack_type,
                                   stacking_skip=stacking_skip)
    return images_, actions_


def load_hdf_data(
    game='breakout',
    dataset=['564_RZ_4602455_Jul-31-14-48-16'],
    data_types=['images', 'actions', 'gazes', 'gazes_fused_noop']):
    """ Loads data from the hdf game file 
    
    Args:
    ----
     game : game to load the data from, directory of game runs
     dataset : game_run to load the data from, a list of game runs.
                dataset=['564_RZ_4602455_Jul-31-14-48-16'].
     data_types : types of data to load from the file a list of data types.
                ['images', 'actions', 'fused_gazes']
    Returns:
    ----
     game_data : a dcit of game_data loaded from hdf5file for the specified game runs
    
    """
    game_file = os.path.join(PROC_DATA_DIR, game + '.hdf5')
    game_h5_file = h5py.File(game_file, 'r')
    game_data = []
    if dataset is -1:
        dataset = list(game_h5_file.keys())
    game_data = {k: [] for k in data_types}

    actions = []
    for game_run in dataset:
        assert game_h5_file.__contains__(game_run), print(
            game_run, "doesn't exist in game", game)
        game_run_data_h5 = game_h5_file[game_run]
        for datum in data_types:
            assert game_run_data_h5.__contains__(datum), print(
                datum, "doesn't exist in game", game, game_run)
            game_data[datum].append(game_run_data_h5[datum][:])
    return game_data


def load_data_iter(
    game=None,
    data_types=['images', 'actions', 'gazes', 'gazes_fused_noop'],
    dataset='combined',
    dataset_exclude=['combined'],
    device=torch.device('cpu'),
    load_type='memory',
    batch_size=32,
    sampler=None):
    """
    Creates a dataset and a data iterator to use in the train loop
    
    Args:
    ----
     game : game to load the data from, directory of game runs
     data_types : types of data to load, contains atleast on of the following
                ['frames/images', 'actions', 'gazes',' gazes_fused_noop']

     dataset : game_run to load the data from, directory of frames and gaze data
     device : device to load the data to cpu or gpu
     load_type : data load type, different types are described below
        'memory' --  'loads everything into memoey if possible else errors
                        fastest option'
        'disk' -- 'Reads from the hdf5 file the specified index every 
                        iteration,slowest'
        'live'  -- 'Directly loads from the interim files and  
                        preprocesss the data'
        'chunked' -- 'Splits the given dataset hdf5 file into 
                        specified chunks and cycles through them in the train loop'
    'batch_size : batch size of the data to iterate over, default 32
    sampler : Type of sampler class to use when sampling the data, defualt None 

    Returns:
    ----
     data_iter : a data iterator with the specified data types that can be looped over
    
    """

    if load_type == 'memory':
        data = load_hdf_data(game=game, dataset=dataset)
        x, y_, _, x_g = data.values()
        x = torch.Tensor(x).squeeze().to(device=device)
        y = torch.LongTensor(y_).squeeze()[:, -1].to(device=device)
        x_g = torch.Tensor(x_g).squeeze().to(device=device)
        dataset = torch.utils.data.TensorDataset(x, y, x_g)
        dataset.labels = y_[0][:, -1]

    elif load_type == 'disk':
        dataset = HDF5TorchDataset(game=game,
                                   data=data_types,
                                   dataset=dataset,
                                   device=device)
    elif load_type == 'live':
        print("prepping and loading data for {},{}".format(game, dataset))
        images_, actions_ = load_action_data(stack=config['STACK_SIZE'],
                                             game=game,
                                             till_ix=-1,
                                             game_run=dataset)

        _, gazes = load_gaze_data(stack=config['STACK_SIZE'],
                                  game=game,
                                  till_ix=-1,
                                  game_run=dataset,
                                  skip_images=True)
        images_ = transform_images(images_, type='torch')
        gazes = fuse_gazes(images_, gazes, gaze_count=1)
        x = images_.to(device=device)
        y = torch.LongTensor(actions_)[:, -1].to(device=device)
        x_g = gazes.to(device=device)
        dataset = torch.utils.data.TensorDataset(x, y, x_g)
        dataset.labels = np.array(actions_)[:, -1]

    elif load_type == 'chunked':
        sampler = None

        dataset = HDF5TorchChunkDataset(game=game,
                                        data_types=data_types,
                                        dataset_exclude=dataset_exclude,
                                        dataset=dataset,
                                        device=device)

    if sampler is None:
        data_iter = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                num_workers=0)
    else:
        data_iter = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                sampler=sampler(dataset))

    return data_iter


class HDF5TorchChunkDataset(data.Dataset):
    def __init__(self,
                 game,
                 data_types=['images', 'actions', 'gazes', 'gazes_fused_noop'],
                 dataset_exclude=['combined'],
                 device=torch.device('cpu'),
                 num_epochs_per_collation=1,
                 num_groups_to_collate=1,dataset=['combined']):
        self.game = game
        self.device = device
        self.dataset = dataset
        self.data_types = data_types
        self.dataset_exclude = dataset_exclude
        self.num_epochs_per_collation = num_epochs_per_collation
        self.num_groups_to_collate = num_groups_to_collate

        self.curr_collation_epoch = 0
        self.curr_collation_data = {}

        hdf5_file = os.path.join(PROC_DATA_DIR, '{}.hdf5'.format(game))
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        groups = list(sorted(set(self.hdf5_file.keys()) - set(self.dataset_exclude),
                   reverse=True))
        if isinstance(self.dataset,list) and 'combined' not in self.dataset:
            groups = self.dataset
        
        if dataset != 'combined':
            groups = self.dataset_exclude
        
        self.groups = cycle(groups)        
        self.group_lens = [
            self.hdf5_file[g]['actions'].len() for g in groups
        ]
        self.total_count = self.num_epochs_per_collation * sum(self.group_lens)

        self.tensors = []
        self.curr_collation = None

        self.__reset_dataset__()

    def __load_data__(self):
        self.curr_collation_data = load_hdf_data(game=self.game,
                                                 dataset=self.curr_collation,
                                                 data_types=self.data_types)

        for dtype in self.curr_collation_data:
            datum = self.curr_collation_data[dtype]
            datum = np.concatenate(datum, axis=0)
            if dtype == 'actions':
                datum = torch.LongTensor(datum).squeeze()[:, -1].to(
                    device=self.device)
            else:
                datum = torch.Tensor(datum).squeeze().to(device=self.device)
            self.curr_collation_data[dtype] = datum
        group_lens = [
            self.curr_collation_data[datum].shape[0]
            for datum in self.curr_collation_data
        ]

        assert len(set(group_lens)) == 1
        self.curr_collation_len = group_lens[0]

    def __reset_dataset__(self):

        self.curr_ix = 0

        if self.curr_collation_epoch == self.num_epochs_per_collation or self.curr_collation is None:
            self.curr_collation_epoch = 0
            # print("Cycling dataset from {}".format(self.curr_collation))
            self.curr_collation = [
                next(self.groups) for _ in range(self.num_groups_to_collate)
            ]
            # print("                   to {}".format(self.curr_collation))
            self.__load_data__()
            if 'actions' in self.curr_collation_data:

                labels = self.curr_collation_data['actions'].cpu().numpy(
                ).copy()
                label_to_count = Counter(labels)
                weights = torch.DoubleTensor(
                    [1.0 / label_to_count[ix] for ix in labels])
                self.sample_ixs = torch.multinomial(weights,
                                                    self.curr_collation_len,
                                                    replacement=True)
            else:
                self.sample_ixs = list(range(self.curr_collation_len))

        self.curr_collation_epoch += 1

    def __len__(self):
        return self.total_count

    def __getitem__(self, ix):
        tensors = []

        if self.curr_ix == self.curr_collation_len:
            self.__reset_dataset__()

        sample_ix = self.sample_ixs[self.curr_ix]

        tensors = {
            dtype: self.curr_collation_data[dtype][sample_ix]
            for dtype in self.data_types
        }

        self.curr_ix += 1

        return tensors


class HDF5TorchDataset(data.Dataset):
    def __init__(self,
                 game,
                 data=['images', 'actions', 'gazes'],
                 dataset='combined',
                 device=torch.device('cpu')):
        hdf5_file = os.path.join(PROC_DATA_DIR, '{}.hdf5'.format(game))
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.data = data
        self.group = self.hdf5_file[dataset]
        self.group_counts = self.group['actions'].len()

        self.tensors = []
        self.labels = self.group['actions'][:, -1]
        self.device = device

    def __len__(self):
        return self.group_counts

    def __getitem__(self, ix):
        tensors = []
        for datum in self.data:
            if datum == 'actions':
                tensors.append(
                    torch.tensor(self.labels[ix]).to(device=self.device))
            else:
                tensors.append(
                    torch.Tensor(self.group[datum][ix]).to(device=self.device))
        return tensors


if __name__ == "__main__":

    ds = HDF5TorchChunkDataset(game='breakout',
                               num_groups_to_collate=2,
                               num_epochs_per_collation=1)

    dl = load_data_iter(game='breakout', batch_size=1, load_type='chunked')

    for x in dl:
        print(x.keys())
