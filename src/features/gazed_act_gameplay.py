import os
import sys
import torch
from tqdm import tqdm
from yaml import safe_load
from src.models.selective_gaze_only import SGAZED_ACTION_SL
from src.models.action_sl import ACTION_SL
from src.data.data_loaders import load_action_data, load_gaze_data
from src.models.cnn_gaze import CNN_GAZE
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gym
from gym.wrappers import FrameStack,Monitor
from src.features.feat_utils import image_transforms
import numpy as np
transform_images = image_transforms()
from src.data.data_loaders import load_hdf_data
from gym.envs.atari import AtariEnv
# pylint: disable=all
from feat_utils import image_transforms, reduce_gaze_stack, draw_figs, fuse_gazes, fuse_gazes_noop,gaze_pdf  # nopep8
import argparse

import cv2

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())
np.random.seed(42)
MODE = 'eval'
BATCH_SIZE = config_data['BATCH_SIZE']
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--game',required=True)
parser.add_argument('--action_cpt',required=True)
parser.add_argument('--gaze_net_cpt',required=True)
parser.add_argument('--episode',default=None)
args = parser.parse_args()
game = args.game
action_cpt = args.action_cpt
gaze_net_cpt = args.gaze_net_cpt
episode = args.episode
# GAZE_TYPE = ["PRED","REAL"]
GAZE_TYPE = "PRED"

if game == 'phoenix':
    dataset_val = ['606_RZ_5215078_Aug-07-16-59-46', '600_RZ_5203429_Aug-07-13-44-39',
                   '598_RZ_5120717_Aug-06-14-53-30', '574_RZ_4682055_Aug-01-12-54-58',
                   '565_RZ_4604537_Jul-31-15-22-57', '550_RZ_4513408_Jul-30-14-04-09',
                   '540_RZ_4425986_Jul-29-13-47-10']
    env_name = 'Phoenix-v0'

elif game == 'asterix':
    dataset_val = ['543_RZ_4430054_Jul-29-14-54-56',
                   '246_RZ_721092_Feb-20-21-52-26','260_RZ_1456515_Mar-01-10-10-36',
                   '534_RZ_4166872_Jul-26-13-49-43','553_RZ_4519853_Jul-30-15-51-37']
    env_name = 'Asterix-v0'

elif game == 'breakout':
    dataset_val = [ '564_RZ_4602455_Jul-31-14-48-16',
                    '527_RZ_4153166_Jul-26-10-00-12']
    env_name = 'Breakout-v0'

elif game == 'freeway':
    dataset_val = ['151_JAW_3358283_Dec-15-11-19-24','157_KM_6307437_Jan-18-14-31-43',
                    '149_JAW_3355334_Dec-15-10-31-51','79_RZ_3074177_Aug-18-11-46-29',
                    '156_KM_6306308_Jan-18-14-13-55']
    env_name = 'Freeway-v0'

elif game == 'name_this_game':
    dataset_val = ['267_RZ_2956617_Mar-18-19-52-47','576_RZ_4685615_Aug-01-13-54-21']                    
    env_name = 'NameThisGame-v0'

elif game == 'space_invaders':
    dataset_val = ['554_RZ_4520643_Jul-30-16-08-32','511_RZ_3988011_Jul-24-12-07-48',
                    '512_RZ_3991738_Jul-24-13-12-15','514_RZ_3993948_Jul-24-13-47-04',
                    '541_RZ_4427259_Jul-29-14-08-29','587_RZ_4775423_Aug-02-14-51-06',
                    '596_RZ_5117737_Aug-06-13-56-16'
                    ]
    env_name = 'SpaceInvaders-v0'

elif game == 'demon_attack':
    env_name = 'DemonAttack-v0'

elif game == 'seaquest':
    env_name = 'Seaquest-v0'

elif game == 'ms_pacman':
    env_name = 'MsPacman-v0'

elif game == 'centipede':
    env_name = 'Centipede-v0'
    
dataset_train = dataset_val = 'combined'  # game_run

device = torch.device('cuda')

data_types = ['images', 'actions', 'gazes_fused_noop']

action_net =SGAZED_ACTION_SL(game=game,
                              data_types=data_types,
                              dataset_train=dataset_train,
                              dataset_train_load_type='chunked',
                              dataset_val=dataset_val,
                              dataset_val_load_type='chunked',
                              device=device,
                              mode=MODE).to(device=device)
action_net.load_model_fn(action_cpt)

gaze_net = CNN_GAZE(game=game,
                            data_types=data_types,
                            dataset_train=dataset_train,
                            dataset_val=dataset_val,
                            dataset_train_load_type='chunked',
                            dataset_val_load_type='chunked',
                            device=device,
                            mode='eval').to(device=device)

gaze_net.epoch = gaze_net_cpt
gaze_net.load_model_fn(gaze_net_cpt)

optimizer = torch.optim.Adadelta(action_net.parameters(), lr=1.0, rho=0.95)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: x*0.95)
lr_scheduler = None
loss_ = torch.nn.CrossEntropyLoss().to(device=device)
env = gym.make(env_name, full_action_space=True,frameskip=1)

env = FrameStack(env, 4)
env = Monitor(env,env_name,force=True)

t_rew = 0
print(env._env_info())
if episode is None:
    start_episode = 0
    end_episode = 30
else:
    start_episode = int(episode)
    end_episode = start_episode+1
for i_episode in range(start_episode,end_episode,1):
    env.seed(i_episode)
    
    observation = env.reset()
    ep_rew = 0
    i =-1
    while True:
        # env.render()
        i+=1

        obs = observation
        observation = torch.stack(
            [transform_images(o).squeeze() for o in observation]).unsqueeze(0).to(device=device)
        xg = gaze_net.infer(observation)
        gaze = torch.exp(xg)
        gaze_true = gaze.squeeze().cpu().data.numpy()

        gmax = np.unravel_index(np.argmax(gaze_true, axis=None), gaze_true.shape)
        gaze_true = np.array(gmax)/84.0

        gaze_true = gaze_pdf([gaze_true[::-1]])
        gazes = []
        for g in gaze:
            g = (g - torch.min(g)) / (torch.max(g) - torch.min(g))
            # g = g / torch.sum(g)
            gazes.append(g)
        gaze = torch.stack(gazes).unsqueeze(1)
        gaze_ = gaze.squeeze().cpu().numpy()
        pile = np.percentile(gaze_,90)
        gaze_ = np.clip(gaze_,pile,1)
        
        # gaze_ = np.array(cv2.resize(gaze_,(160,210))*255,dtype=np.uint8)
        # gaze_ = cv2.applyColorMap(gaze_,cv2.COLORMAP_INFERNO)
        # gaze_ = obs[-1]+gaze_


        # cv2.imshow("gaze_pred_normalized",gaze_)
        # cv2.waitKey(0)

        gaze_ = gaze[0][0].cpu().numpy()
        
        gaze_ = np.array(cv2.resize(gaze_,(160,210))*255,dtype=np.uint8)
        gaze_ = cv2.applyColorMap(gaze_,cv2.COLORMAP_TURBO)
        obs = cv2.resize(obs[-1],(160,210))

        obs = cv2.cvtColor(obs,cv2.COLOR_RGB2BGR)
        gaze_ = cv2.addWeighted(gaze_,0.25,obs,0.5,0)*2

        gaze_true = (gaze_true - np.min(gaze_true)) / (np.max(gaze_true) - np.min(gaze_true))

        gaze_true = np.array(cv2.resize(gaze_true,(160,210))*255,dtype=np.uint8)
        gaze_true = cv2.applyColorMap(gaze_true,cv2.COLORMAP_TURBO)
        
        observation = observation[:,-1].unsqueeze(1)
        gaze = gaze * observation
        action = action_net.infer(observation,gaze).data.item()
        
        
        observation, reward, done, info = env.step(action)

        ep_rew += reward
        if done:
            # print("Episode finished after {} timesteps".format(t + 1))
            break
    t_rew += ep_rew
    print("Episode {} reward {}".format(i_episode, ep_rew))
    print("Ave Episode {} reward {}".format(i_episode, t_rew/(i_episode+1)))
    

print("Mean all Episode {} reward {}".format(i_episode, t_rew / (i_episode+1)))
env.close()
