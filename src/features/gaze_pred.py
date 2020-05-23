import os
import sys
import torch
from tqdm import tqdm
from yaml import safe_load
from src.models.cnn_gaze import CNN_GAZE
from src.data.data_loaders import load_action_data, load_gaze_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data.data_loaders import load_hdf_data
import numpy as np
from feat_utils import gaze_pdf

# pylint: disable=all
from feat_utils import image_transforms, reduce_gaze_stack, draw_figs, fuse_gazes, fuse_gazes_noop  # nopep8

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

MODE = 'eval'
BATCH_SIZE = config_data['BATCH_SIZE']

game = 'freeway'
# game = 'name_this_game'
dataset_train = dataset_val = 'combined'  #game_run
dataset_val = ['79_RZ_3074177_Aug-18-11-46-29']
dataset_val = ['55_RZ_2464601_Aug-11-10-18-09']
device = torch.device('cuda')

data_types = ['images', 'gazes']

gaze_net = CNN_GAZE(game=game,
                    data_types=data_types,
                    dataset_train=dataset_train,
                    dataset_val=dataset_val,
                    dataset_train_load_type='chunked',
                    dataset_val_load_type='chunked',
                    device=device,
                    mode=MODE).to(device=device)

optimizer = torch.optim.Adadelta(gaze_net.parameters(), lr=1.0, rho=0.95)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: x*0.95)
lr_scheduler = None
loss_ = torch.nn.KLDivLoss(reduction='batchmean')

if MODE=='eval':
    curr_group_data = load_hdf_data(game=game,
                                        dataset=dataset_val,
                                        data_types=['images','gazes'],
                                        )
    
    x, y = curr_group_data.values()
    x = x[0]
    y = y[0]

    image_ = x[205]
    gaze_ =  y[205]

    for cpt in tqdm(range(28, 29, 1)):
        gaze_net.epoch = cpt
        gaze_net.load_model_fn(cpt)
        smax = gaze_net.infer(
           torch.Tensor(image_).to(device=device).unsqueeze(0)).squeeze().cpu().data.numpy()

        g_max = np.array(g_max)/84.0
        gaze_max = np.array(gaze_max)/84.0
        # smax = gaze_pdf([g_max])
        # gaze_ = gaze_pdf([gaze_max])
        pile = np.percentile(smax,90)
        smax = np.clip(smax,pile,1)
        smax = (smax-np.min(smax))/(np.max(smax)-np.min(smax))
        smax = smax/np.sum(smax)

        draw_figs(x_var=smax, gazes=gaze_*255)

else:
    gaze_net.train_loop(optimizer, lr_scheduler, loss_, batch_size=BATCH_SIZE)
