import os
import sys
import torch
from tqdm import tqdm
from yaml import safe_load
from src.models.action_sl import ACTION_SL
from src.data.data_loaders import load_action_data

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=all
from feat_utils import image_transforms, reduce_gaze_stack, draw_figs  # nopep8

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

INFER = False
game = 'breakout'

# VALID_ACTIONS = config_data['VALID_ACTIONS'][game]
# num_actions = len(VALID_ACTIONS)
device = torch.device('cuda')

action_net = ACTION_SL().to(device=device)

optimizer = torch.optim.Adadelta(action_net.parameters(), lr=1.0, rho=0.95)

# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: x*0.95)

lr_scheduler = None
loss_ = torch.nn.CrossEntropyLoss()

# single game run data load, default loads only 10 data points
images, actions = load_action_data(stack=4,
                                   till_ix=-1,
                                   game_run='527_RZ_4153166_Jul-26-10-00-12')
assert len(images) == len(actions)

transforms_ = image_transforms(image_size=(84, 84))

# image crop and other transforms
images_ = [
    torch.stack([transforms_(image_).squeeze() for image_ in image_stack])
    for image_stack in images
]
# import numpy as np
# from collections import Counter
# print(Counter(np.array(actions).flatten()))
actions_ = [action_stack[-1] for action_stack in actions]
# print(Counter(np.array(actions_).flatten()))
# actions_ = [np.max(action_stack) for action_stack in actions]
# print(Counter(np.array(actions_).flatten()))

assert len(images_) == len(actions_)

x_variable = torch.stack(images_).to(device=device)
y_variable = torch.LongTensor(actions_).to(device=device)
batch_size = min(32, x_variable.shape[0])

if INFER:
    test_ix = 0
    image_ = images[test_ix]
    action_ = actions_[test_ix]
    for cpt in tqdm(range(700, 800, 100)):
        action_pred = action_net.infer(cpt, x_variable[test_ix].unsqueeze(0))
        print(action_, action_pred)
else:
    action_net.train_loop(optimizer, lr_scheduler, loss_, x_variable,
                          y_variable, batch_size)
