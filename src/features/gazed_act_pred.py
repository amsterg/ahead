import os
import sys
import torch
from tqdm import tqdm
from yaml import safe_load
from src.models.gazed_action_sl import GAZED_ACTION_SL
from src.data.data_loaders import load_action_data, load_gaze_data
from src.models.cnn_gaze import CNN_GAZE
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data.data_loaders import load_hdf_data

# pylint: disable=all
from feat_utils import image_transforms, reduce_gaze_stack, draw_figs, fuse_gazes, fuse_gazes_noop  # nopep8

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

INFER = False
BATCH_SIZE = config_data['BATCH_SIZE']
# GAZE_TYPE = ["PRED","REAL"]
GAZE_TYPE = "REAL"

# only valid if GAZE_TYPE is PRED
GAZE_PRED_TYPE = "CNN"

game = 'breakout'
game = 'name_this_game'
dataset_train = '198_RZ_3877709_Dec-03-16-56-11'  #game_run
dataset_val = '564_RZ_4602455_Jul-31-14-48-16'
dataset_train = dataset_val = '576_RZ_4685615_Aug-01-13-54-21'
device = torch.device('cuda')

if GAZE_TYPE == "PRED":
    data = ['images', 'actions']
else:
    data = ['images', 'actions', 'fused_gazes']

action_net = GAZED_ACTION_SL(game=game,
                             data=data,
                             dataset_train=dataset_train,
                             dataset_val=dataset_val,
                             device=device).to(device=device)

optimizer = torch.optim.Adadelta(action_net.parameters(), lr=1.0, rho=0.95)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: x*0.95)
lr_scheduler = None
loss_ = torch.nn.CrossEntropyLoss().to(device=device)

if INFER:
    test_ix = 0
    image_ = images[test_ix]
    action_ = actions_[test_ix]
    for cpt in tqdm(range(700, 800, 100)):
        action_net.epoch = cpt
        action_pred = action_net.infer(image_.unsqueeze(0))
        print(action_, action_pred)
else:
    if GAZE_TYPE == "PRED":
        gaze_net = CNN_GAZE()
        gaze_net.epoch = 3800
        action_net.train_loop(optimizer,
                              lr_scheduler,
                              loss_,
                              batch_size=BATCH_SIZE,
                              gaze_pred=gaze_net)

    else:
        action_net.train_loop(optimizer,
                              lr_scheduler,
                              loss_,
                              batch_size=BATCH_SIZE)
