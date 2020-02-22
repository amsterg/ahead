
import os
import sys
import torch
from tqdm import tqdm
from yaml import safe_load
from src.models.gazed_action_sl import GAZED_ACTION_SL
from src.data.load_interim import load_action_data, load_gaze_data
from src.models.cnn_gaze import CNN_GAZE
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=all
from feat_utils import image_transforms, reduce_gaze_stack, draw_figs, fuse_gazes  # nopep8

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

INFER = False

# GAZE_TYPE = ["PRED","REAL"]
GAZE_TYPE = "PRED"

# only valid if GAZE_TYPE is PRED
GAZE_PRED_TYPE = "CNN"

DATA_COUNT = 10
game = 'breakout'

transforms_ = image_transforms(image_size=(84, 84))


def prep_tensors():

    images, actions = load_action_data(stack=4, till_ix=DATA_COUNT)

    images_ = [torch.stack([transforms_(image_).squeeze()
                            for image_ in image_stack]) for image_stack in images]
    actions_ = [action_stack[-1] for action_stack in actions]
    x_variable = torch.stack(images_)
    y_variable = torch.LongTensor(actions_)
    batch_size = min(32, x_variable.shape[0])

    if GAZE_TYPE == "PRED":
        xg_variable = []
        return x_variable, xg_variable, y_variable, batch_size
    else:
        _, gazes = load_gaze_data(
            stack=4, till_ix=DATA_COUNT, skip_images=True)

    assert len(images) == len(actions) == len(
        gazes), print(len(images), len(actions), len(gazes))

    fused_gazes = fuse_gazes(images_, gazes)

    assert len(images_) == len(actions_) == len(fused_gazes)

    xg_variable = fused_gazes

    return x_variable, xg_variable, y_variable, batch_size


action_net = GAZED_ACTION_SL()
optimizer = torch.optim.Adadelta(
    action_net.parameters(), lr=1.0, rho=0.95)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: x*0.95)
lr_scheduler = None
loss_ = torch.nn.CrossEntropyLoss()

x_variable, xg_variable, y_variable, batch_size = prep_tensors()

if INFER:
    test_ix = 0
    image_ = images[test_ix]
    action_ = actions_[test_ix]
    for cpt in tqdm(range(700, 800, 100)):
        action_net.epoch = cpt
        action_pred = action_net.infer(x_variable[test_ix].unsqueeze(0))
        print(action_, action_pred)
else:
    if isinstance(xg_variable, list) or GAZE_TYPE == "PRED":
        gaze_net = CNN_GAZE()
        gaze_net.epoch = 3800
        action_net.train_loop(optimizer, lr_scheduler, loss_,
                              x_variable, y_variable, batch_size=batch_size, gaze_pred=gaze_net)

    else:
        action_net.train_loop(optimizer, lr_scheduler, loss_, x_variable,
                              y_variable, xg_variable, batch_size=batch_size)
