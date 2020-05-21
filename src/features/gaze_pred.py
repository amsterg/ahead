import os
import sys
import torch
from tqdm import tqdm
from yaml import safe_load
from src.models.cnn_gaze import CNN_GAZE
from src.data.data_loaders import load_action_data, load_gaze_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data.data_loaders import load_hdf_data

# pylint: disable=all
from feat_utils import image_transforms, reduce_gaze_stack, draw_figs, fuse_gazes, fuse_gazes_noop  # nopep8

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

INFER = False
BATCH_SIZE = config_data['BATCH_SIZE']


game = 'asterix'
# game = 'name_this_game'
dataset_train = dataset_val = 'combined'  #game_run
# dataset_val = '564_RZ_4602455_Jul-31-14-48-16'
device = torch.device('cuda')

data = ['images', 'gazes']

gaze_net = CNN_GAZE(game=game,
                             data=data,
                             dataset_train=dataset_train,
                             dataset_val=dataset_val,
                             dataset_train_load_type='chunked',
                             dataset_val_load_type='memory',
                             device=device).to(device=device)

optimizer = torch.optim.Adadelta(gaze_net.parameters(), lr=1.0, rho=0.95)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: x*0.95)
lr_scheduler = None
loss_ = torch.nn.KLDivLoss(reduction='batchmean')

if INFER:
    test_ix = 0
    image_ = images[test_ix]
    gaze_ = gazes[test_ix]
    for cpt in tqdm(range(1000, 1020, 10)):
        gaze_net.epoch = cpt
        smax = gaze_net.infer(
            x_variable[test_ix].unsqueeze(0)).data.numpy()
        draw_figs(x_var=smax[0], gazes=gazes_[test_ix].numpy())

else:
    gaze_net.train_loop(optimizer,
                            lr_scheduler,
                            loss_,
                            batch_size=BATCH_SIZE)
