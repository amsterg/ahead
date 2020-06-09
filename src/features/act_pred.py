from src.data.data_loaders import load_hdf_data
import os
import sys
import torch
from tqdm import tqdm
from yaml import safe_load
from src.models.action_sl import ACTION_SL
from src.data.data_loaders import load_action_data, load_gaze_data
from src.models.cnn_gaze import CNN_GAZE
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=all
from feat_utils import image_transforms, reduce_gaze_stack, draw_figs, fuse_gazes, fuse_gazes_noop  # nopep8

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

MODE = 'train'
BATCH_SIZE = config_data['BATCH_SIZE']
# GAZE_TYPE = ["PRED","REAL"]
GAZE_TYPE = "PRED"

game = 'phoenix'
dataset_train = ['combined']
dataset_val = ['540_RZ_4425986_Jul-29-13-47-10']

device = torch.device('cuda')

data_types = ['images', 'actions', 'gazes']

action_net = ACTION_SL(game=game,
                             data_types=data_types,
                             dataset_train=dataset_train,
                             dataset_train_load_type='chunked',
                             dataset_val=dataset_val,
                             dataset_val_load_type='chunked',
                             device=device,
                             mode=MODE).to(device=device)



optimizer = torch.optim.Adadelta(action_net.parameters(), lr=1.0, rho=0.9)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: x*0.95)
lr_scheduler = None
loss_ = torch.nn.CrossEntropyLoss().to(device=device)

# action_net.load_model_fn(19)
# optimizer.load_state_dict()

if MODE == 'eval':
        curr_group_data = load_hdf_data(game=game,
                                        dataset=dataset_val,
                                        data_types=['images','actions'],
                                        )
    
        x, y = curr_group_data.values()
        x = x[0]
        y = y[0]

        image_ = x[204]
        image_ = torch.Tensor(image_).to(device=device).unsqueeze(0)
        action =  y[205]

        acts = action_net.infer(image_)
        acts = acts#.data.item()

    

else:
    action_net.train_loop(optimizer,
                              lr_scheduler,
                              loss_,
                              batch_size=BATCH_SIZE)
