
import os
import sys
import torch
from tqdm import tqdm
from yaml import safe_load
from src.models.cnn_gaze import CNN_GAZE
from src.data.load_interim import load_gaze_data

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=all
from utils import image_transforms, reduce_gaze_stack, draw_figs  # nopep8

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

INFER = False
cnn_gaze_net = CNN_GAZE()
optimizer = torch.optim.Adadelta(
    cnn_gaze_net.parameters(), lr=1.0, rho=0.95)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: x*0.95)
lr_scheduler = None
loss_ = torch.nn.KLDivLoss(reduction='batchmean')

# single game run data load
images, gazes = load_gaze_data(stack=4)
assert len(images) == len(gazes)

transforms_ = image_transforms(image_size=(84, 84))

# image crop and oher transforms
images_ = [torch.stack([transforms_(image_).squeeze()
                        for image_ in image_stack]) for image_stack in images]

# transform gazes to tensors with clustering
# gazes_ = [torch.stack([torch.Tensor(gaze_clusters(gaze_, num_clusters))
#                        for gaze_ in gaze_stack]) for gaze_stack in gazes]

# transform gazes to tensors as a weighted distirbution
# gazes_ = [torch.stack([torch.Tensor(gaze_pdf(gaze_))
#                        for gaze_ in gaze_stack]) for gaze_stack in gazes]

# transform gazes to tensors as a weighted distirbution, and reduce stack
gazes_ = [reduce_gaze_stack(gaze_stack) for gaze_stack in gazes]

assert len(images_) == len(gazes_)

x_variable = torch.stack(images_)
y_variable = torch.stack(gazes_)
batch_size = min(32, x_variable.shape[0])

if INFER:
    test_ix = 0
    image_ = images[test_ix]
    gaze_ = gazes[test_ix]
    for cpt in tqdm(range(1000, 1020, 10)):
        smax = cnn_gaze_net.infer(cpt, x_variable[test_ix].unsqueeze(0))
        draw_figs(x_var=smax[0], gazes=gazes_[test_ix].numpy())
else:
    cnn_gaze_net.train_loop(optimizer, lr_scheduler, loss_, x_variable,
                            y_variable, batch_size)
