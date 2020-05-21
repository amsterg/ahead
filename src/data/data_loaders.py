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
                   stacking_skip=0,
                   from_ix=0,
                   till_ix=10,
                   game='breakout',
                   game_run='198_RZ_3877709_Dec-03-16-56-11',
                   skip_images=False):
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
                     stacking_skip=0,
                     from_ix=0,
                     till_ix=10,
                     game='breakout',
                     game_run='198_RZ_3877709_Dec-03-16-56-11'):
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


def load_game_action_data(stack=1,
                          stack_type='',
                          stacking_skip=0,
                          from_ix=0,
                          till_ix=-1,
                          game='breakout'):
    game_dir = os.path.join(INTERIM_DATA_DIR, game)
    game_runs = os.listdir(game_dir)
    images = []
    actions = []
    gaze_out_h5_file = os.path.join(PROC_DATA_DIR, game + '.hdf5')
    gaze_h5_file = h5py.File(gaze_out_h5_file, 'w')

    for game_run in tqdm(game_runs):
        print(game, game_run)
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

        gazes = fuse_gazes_noop(images_,
                                gazes,
                                actions_,
                                gaze_count=1,
                                fuse_type='stack',
                                fuse_val=1)

        images_ = images_.numpy()
        gazes = gazes.numpy()

        group = gaze_h5_file.create_group(game_run)
        group.create_dataset('images', data=images_, compression="gzip")
        group.create_dataset('actions', data=actions_, compression="gzip")
        group.create_dataset('fused_gazes', data=gazes, compression="gzip")

        # print(game, game_runs,
        #       np.array(images_).shape,
        #       np.array(actions_).shape,
        #       np.array(gazes).shape)
        #   np.array(gazes).shape)
        # exit()

        del gazes, images_, actions_

    gaze_h5_file.close()


def load_hdf_data(game='breakout',
                  dataset=['564_RZ_4602455_Jul-31-14-48-16'],
                  data=['images', 'actions', 'fused_gazes'],
                  data_type='gazed_images'):
    game_file = os.path.join(PROC_DATA_DIR,
                             game + '_{}.hdf5'.format(data_type))
    game_h5_file = h5py.File(game_file, 'r')
    game_data = []
    if dataset is -1:
        dataset = list(game_h5_file.keys())
    game_data = {k: [] for k in data}

    actions = []
    for game_run in dataset:
        # print(game_run)
        assert game_h5_file.__contains__(game_run), print(
            game_run, "doesn't exist in game", game)
        game_run_data_h5 = game_h5_file[game_run]

        for datum in data:
            assert game_run_data_h5.__contains__(datum), print(
                datum, "doesn't exist in game", game, game_run)
            game_data[datum].append(game_run_data_h5[datum][:])
    return game_data


def load_data_iter(game=None,
                   data=['images', 'actions', 'fused_gazes'],
                   dataset='combined',
                   device=torch.device('cpu'),
                   load_type='memory',
                   batch_size=32,
                   sampler=None):
    """
    Summary:
    
    Args:
    
    Returns:
    
    """

    if load_type == 'memory':
        data = load_hdf_data(game=game, dataset=[dataset])
        x, y_, x_g = data.values()
        x = torch.Tensor(x).squeeze().to(device=device)
        y = torch.LongTensor(y_).squeeze()[:, -1].to(device=device)
        x_g = torch.Tensor(x_g).squeeze().to(device=device)
        dataset = torch.utils.data.TensorDataset(x, y, x_g)
        dataset.labels = y_[0][:, -1]

    elif load_type == 'disk':
        dataset = HDF5TorchDataset(game=game,
                                   data=data,
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

        if 'fused_gazes' in data:

            dataset = HDF5TorchChunkDataset(game=game,
                                            data=data,
                                            device=device)
        else:
            dataset = HDF5TorchChunkGazeDataset(game=game,
                                                data=data,
                                                device=torch.device('cpu'))

    if sampler is None:
        data_iter = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0)
    else:
        data_iter = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                sampler=sampler(dataset))

    return data_iter


class HDF5TorchChunkDataset(data.Dataset):
    def __init__(self,
                 game,
                 data=['images', 'actions', 'fused_gazes'],
                 device=torch.device('cpu')):
        hdf5_file = os.path.join(PROC_DATA_DIR, '{}_gazed_images.hdf5'.format(game))
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.data = data
        self.groups = cycle(
            sorted(set(self.hdf5_file.keys()) - set(['combined']),
                   reverse=True))
        self.epochs_per_group = 1
        self.curr_group_epoch = 0
        self.game = game
        self.tensors = []
        self.device = device
        self.curr_group = None
        self.group_lens = [
            self.hdf5_file[g]['actions'].len()
            for g in list(set(self.hdf5_file.keys()) - set(['combined']))
        ]
        self.num_groups_to_collate = 1
        if self.num_groups_to_collate == 1:
            self.total_count = self.epochs_per_group*sum(self.group_lens)
        else:
            self.total_count = sum(self.group_lens)
        self.x = None
        self.x_g = None
        self.y_ = None
        self.__reset_dataset__()

    def __load_data__(self):
        curr_group_data = load_hdf_data(game=self.game,
                                        dataset=self.curr_group,
                                        data=self.data,
                                        data_type='gazed_images')
        del self.x
        del self.y_
        del self.x_g
        self.x, self.y_, self.x_g = curr_group_data.values()
        self.x = np.concatenate(self.x,axis=0)
        self.y_ = np.concatenate(self.y_,axis=0)
        self.x_g = np.concatenate(self.x_g,axis=0)

        self.x = torch.Tensor(self.x).squeeze().to(device=self.device)
        self.y = torch.LongTensor(self.y_).squeeze()[:,
                                                     -1].to(device=self.device)
        self.x_g = torch.Tensor(self.x_g).squeeze().to(device=self.device)
        
        self.y_ = self.y_[:, -1]
        self.curr_group_len = self.x.shape[0]

    def __reset_dataset__(self):

        self.curr_ix = 0
        if self.num_groups_to_collate > 1:  #only 1 epoch per collated group
            print("Changing dataset from {}".format(self.curr_group))
            self.curr_group = [
                next(self.groups) for _ in range(self.num_groups_to_collate)
            ]
            print("                   to {}".format(self.curr_group))

            self.__load_data__()
            label_to_count = Counter(self.y_)
            weights = torch.DoubleTensor(
                [1.0 / label_to_count[ix] for ix in self.y_])
            self.sample_ixs = torch.multinomial(weights,
                                                self.curr_group_len,
                                                replacement=True)
        else:
            if self.curr_group_epoch == self.epochs_per_group or self.curr_group is None:
                print("Changing dataset from {}".format(self.curr_group))
                self.curr_group = [next(self.groups)]
                print("                   to {}".format(self.curr_group))

                self.__load_data__()
                label_to_count = Counter(self.y_)
                weights = torch.DoubleTensor(
                    [1.0 / label_to_count[ix] for ix in self.y_])
                self.sample_ixs = torch.multinomial(weights,
                                                    self.curr_group_len,
                                                    replacement=True)
                
            self.curr_group_epoch += 1

    def __len__(self):
        return self.total_count

    def __getitem__(self, ix):
        tensors = []

        if self.curr_ix == self.curr_group_len:
            self.__reset_dataset__()

        sample_ix = self.sample_ixs[self.curr_ix]

        tensors = [self.x[sample_ix], self.y[sample_ix], self.x_g[sample_ix]]

        self.curr_ix += 1

        return tensors


class HDF5TorchChunkGazeDataset(data.Dataset):
    def __init__(self,
                 game,
                 data=['images', 'gazes'],
                 device=torch.device('cpu')):
        hdf5_file = os.path.join(PROC_DATA_DIR, '{}_gazes.hdf5'.format(game))
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.data = data
        self.groups = cycle(
            sorted(set(self.hdf5_file.keys()) - set(['combined']),reverse=True))
        self.game = game
        self.epochs_per_group = 3
        self.curr_group_epoch = 0
        self.tensors = []
        self.device = device
        self.curr_group = None
        self.x = None
        self.y = None
        self.group_lens = [
            self.hdf5_file[g]['gazes'].len()
            for g in list(set(self.hdf5_file.keys()) - set(['combined']))
        ]
        self.num_groups_to_collate = 3
        if self.num_groups_to_collate == 1:
            self.total_count = self.epochs_per_group*sum(self.group_lens)
        else:
            self.total_count = sum(self.group_lens)

        self.__reset_dataset__()

    def __load_data__(self):
        curr_group_data = load_hdf_data(game=self.game,
                                        dataset=self.curr_group,
                                        data=self.data,
                                        data_type='gazes')
        del self.x
        del self.y
        self.x, self.y = curr_group_data.values()
        self.x = np.concatenate(self.x,axis=0)
        self.y = np.concatenate(self.y,axis=0)
        self.x = torch.Tensor(self.x).squeeze().to(device=self.device)
        self.y = torch.Tensor(self.y).squeeze().to(device=self.device)
        self.curr_group_len = self.x.shape[0]

    def __reset_dataset__(self):
        self.curr_ix = 0
        if self.num_groups_to_collate > 1:  #only 1 epoch per collated group
            print("Changing dataset from {}".format(self.curr_group))
            self.curr_group = [
                next(self.groups) for _ in range(self.num_groups_to_collate)
            ]
            print("                   to {}".format(self.curr_group))

            self.__load_data__()
        else:

            if self.curr_group_epoch == self.epochs_per_group or self.curr_group is None:
                self.curr_group_epoch = 0
                print("Changing dataset from {}".format(self.curr_group))
                self.curr_group = [next(self.groups)]
                print("                   to {}".format(self.curr_group))
                self.__load_data__()
            self.curr_group_epoch += 1

    def __len__(self):
        return self.total_count

    def __getitem__(self, ix):
        tensors = []

        if self.curr_ix == self.curr_group_len:
            self.__reset_dataset__()
            # self.curr_ix = 0

        sample_ix = self.curr_ix
        # sample_ix = np.random.randint(0,self.curr_group_len)
        # sample_ix = ix

        tensors = [self.x[sample_ix], self.y[sample_ix]]

        self.curr_ix += 1

        return tensors


class HDF5TorchDataset(data.Dataset):
    def __init__(self,
                 game,
                 data=['images', 'actions', 'fused_gazes'],
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
                # self.tensors.append(self.group[datum][ix][-1])
                tensors.append(
                    torch.tensor(self.labels[ix]).to(device=self.device))
            else:
                tensors.append(
                    torch.Tensor(self.group[datum][ix]).to(device=self.device))
        return tensors


if __name__ == "__main__":
    # load_gaze_data()
    # load_action_data(stack=4)
    # load_game_action_data(stack=4)
    # hdf5t = HDF5TorchDataset('breakout')
    # dl = data.DataLoader(hdf5t, 32, sampler=ImbalancedDatasetSampler(hdf5t))
    ds = HDF5TorchChunkDataset(game='breakout')
    dl = data.DataLoader(ds, 3, shuffle=True)

    for x in dl:
        x, y, x_g = x
    dl = load_data_iter(game='breakout',
                        dataset='combined',
                        batch_size=1,
                        load_type='disk')

    arrivals = []
    # dl = data.DataLoader(hdf5t, 32, shuffle=True)
    while True:

        for x in dl:
            #     # print(hdf5t.tensors)
            x, y, x_g = x
            print(x.shape)
            print(y.shape)
            print(x_g.shape)
            arrivals.append(x)
            # print(x)
            # print(y.shape)
            break

        # print(z.s
        break
    # print(np.hstack(arrivals))
    # break
