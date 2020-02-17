import pandas as pd
import os
from yaml import safe_load
import csv
from collections import OrderedDict
import cv2
import numpy as np
from tqdm import tqdm
from subprocess import call
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

RAW_DATA_DIR = config_data['RAW_DATA_DIR']
PROC_DATA_DIR = config_data['PROC_DATA_DIR']
print(RAW_DATA_DIR)
with open(os.path.join(RAW_DATA_DIR, 'action_enums.txt'), 'r') as f:
    ACTIONS_ENUM = f.read()

# game_0 = 'breakout'
game = 'breakout'
VALID_ACTIONS = config_data['VALID_ACTIONS'][game]
print(game, VALID_ACTIONS)
# game_0_run_0
game_runs = [
    entry.split('.txt')[0]
    for entry in os.listdir(os.path.join(RAW_DATA_DIR, game))
    if entry.__contains__('.txt')
]

game_run = game_runs[0]
game_run = '198_RZ_3877709_Dec-03-16-56-11'
print(game_run)
game_run_frames = OrderedDict({
    int(entry.split('_')[-1].split('.png')[0]): entry
    for entry in os.listdir(
        os.path.join(os.path.join(RAW_DATA_DIR, game), game_run))
    if entry.__contains__('.png')
})

game_run_data = []
with open(
        os.path.join(os.path.join(RAW_DATA_DIR, game), game_run) + '.txt',
        'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        game_run_data.append(row)

# print(game_run_data)
header = game_run_data[0]
game_run_data = game_run_data[1:]
game_run_data_mod = []


for tstep in game_run_data:
    tstep_ = []
    tstep_ = tstep[:len(header) - 1]
    if 'null' in tstep_:
        tstep_[1:] = game_run_data_mod[-1][1:len(header) - 1]
        assert len(tstep_) == len(header) - 1, print(tstep_, header,
                                                     len(tstep_), len(header))

    gaze_data = tstep[len(header) - 1:]
    if len(gaze_data) == 1 and gaze_data[0] == 'null':

        gaze_data = game_run_data_mod[-1][len(header) - 1]
        gaze_data_ = gaze_data
        assert int(len(gaze_data) / len(gaze_data_)) == 1.0, print(
            len(gaze_data), len(gaze_data_))
    else:
        gaze_data_ = [
            gaze_data[ix:ix + 2] for ix in range(0,
                                                 len(gaze_data) - 1, 2)
        ]
        assert int(len(gaze_data) / len(gaze_data_)) == 2.0, print(
            len(gaze_data), len(gaze_data_))
    tstep_.append(gaze_data_)
    assert len(tstep_) == len(header)
    game_run_data_mod.append(tstep_)

game_run_data_mod_df = pd.DataFrame(game_run_data_mod, columns=header)
game_run_data_mod_df['action'] = game_run_data_mod_df['action'].apply(
    lambda x: 0 if x not in VALID_ACTIONS else x)

frame_ids = game_run_data_mod_df['frame_id']
assert len(frame_ids) == len(game_run_frames), print(len(frame_ids),
                                                     len(game_run_frames))
proc_writ_dir = os.path.join(os.path.join(PROC_DATA_DIR, game),
                             game_run) + '_wgz'
img_writ_dir = os.path.join(proc_writ_dir, 'images/')
img_diff_writ_dir = os.path.join(proc_writ_dir, 'images_diff/')
if not os.path.exists(img_writ_dir):
    os.makedirs(img_writ_dir)
if not os.path.exists(img_diff_writ_dir):
    os.makedirs(img_diff_writ_dir)
game_run_data_mod_df.to_csv('{}/processed_game_data'.format(proc_writ_dir))

color_map = [
    [255, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
]
frame_ix = None
frame_ixp1 = None
frame_diff = None
wix = -1
frame_sum_ot = 0
game_actions = list(game_run_data_mod_df['action'])

game_actions += [0 for _ in range(10)]
for fid in tqdm(frame_ids.index):
    wix += 1
    assert game_run_data_mod_df.iloc[fid]['frame_id'] == game_run_frames[
        fid + 1].split('.png')[0]
    frame_abs_name = os.path.join(
        os.path.join(os.path.join(RAW_DATA_DIR, game), game_run),
        game_run_frames[fid + 1])
    frame_img_data = cv2.imread(frame_abs_name)
    frame_img_data_np = np.array(frame_img_data)
    frame_img_data = frame_img_data[20:, :]
    frame_img_data = cv2.cvtColor(frame_img_data, cv2.COLOR_BGR2GRAY)
    # canny_edges = cv2.Canny(
    #     imgray,
    #     220,
    #     250,
    # )
    # ret, thresh = cv2.threshold(canny_edges, 10, 20, 0)
    # contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
    #                                   cv2.CHAIN_APPROX_SIMPLE)
    # pad = 1
    # for cnt in contours:

    #     x, y, w, h = cv2.boundingRect(cnt)

    #     frame_img_data = cv2.rectangle(frame_img_data, (x - pad, y - pad),
    #                                    (x + w + pad, y + h + pad), (0, 255, 0),
    #                                    1)
    # frame_img_data = cv2.drawContours(frame_img_data, contours, -1,
    #                                   (0, 255, 0))

    frame_img_data_orig = frame_img_data.copy()
    for ix in range(len(game_run_data_mod_df.iloc[fid]['gaze_positions'])):
        opacity = (0.1 * ix) / len(
            game_run_data_mod_df.iloc[fid]['gaze_positions'])
        gpts = game_run_data_mod_df.iloc[fid]['gaze_positions'][ix]
        gpt = min(int(float(gpts[1])), frame_img_data.shape[0] - 1), min(
            int(float(gpts[0])), frame_img_data.shape[1] - 1)

        frame_img_data_np[gpt] = [255, 255, 255]
        frame_img_data_olay = frame_img_data.copy()
        cv2.circle(frame_img_data_olay, gpt[::-1], 6, (102, 0, 204), -1, 16)
        # frame_img_data_olay -= frame_img_data
        # cv2.GaussianBlur(frame_img_data_olay, (5, 5), 0)
        # cv2.circle(frame_img_data_olay, gpt[::-1], 3, (255, 255, 255), 1)
        # frame_img_data = 0.0*frame_img_data + 1.0 * frame_img_data_olay
        # frame_img_data /= 255.0
        # cv2.GaussianBlur(frame_img_data, (5, 5), 10)
        cv2.addWeighted(frame_img_data_olay, opacity, frame_img_data,
                        1 - opacity, 0, frame_img_data)
        # print(gpt)
        # frame_img_data[gpt] = color_map[ix % len(color_map)]
        # cv2.imshow('Frame {}'.format(fid), frame_img_data)
        # cv2.imshow('Frame_np {}'.format(fid), frame_img_data_np)
        # cv2.waitKey()
        # break
    # opacity = 0.5
    # cv2.addWeighted(frame_img_data, opacity, frame_img_data_orig,
    #                         1-opacity, 0, frame_img_data)

    gpts = np.array(game_run_data_mod_df.iloc[fid]['gaze_positions']).astype(
        np.float).astype(np.int)

    # ax = sns.regplot(x=gpts[:,0], y=gpts[:,1])
    x, y = np.mgrid[0:frame_img_data.shape[1]:1, 0:frame_img_data.shape[0]:1]
    # x = x[::-1]
    # y = y[::-1]
    pos = np.dstack((x, y))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    pdfs = []
    for gpt in gpts:
        rv = multivariate_normal(mean=gpt, cov=5)
        pdfs.append(rv.pdf(pos))
        # plt.contourf(x, y, rv.pdf(pos),alpha=0.1)
        # break
    # ws = np.ones(range(len(pdfs)),1)

    wpdf = np.sum(pdfs, axis=0)
    # print(wpdf.shape)
    plt.contourf(x, y, wpdf)
    plt.ylim(plt.ylim()[::-1])
    plt.close()

    # sns.scatterplot(x=gpts[:,0], y=gpts[:,1])
    # ax = sns.distributions.kdeplot(
    #     pd.DataFrame({'x': gpts[:, 0], 'y': gpts[:, 1]}))
    # ax.set_title(len(gpts))
    # plt.show()
    # plt.show()

    # cv2.circle(frame_img_data, tuple(gpt[::-1]), 6, [255, 255, 255], 1, 16)

    # frame_sum_ot += frame_img_data

    # if wix % 2 == 0:
    #     frame_ix = frame_img_data
    # else:
    #     frame_ixp1 = frame_img_data
    # if frame_ix is not None and frame_ixp1 is not None:
    #     # assert (frame_ix != frame_ixp1).all()
    #     if frame_diff is not None:
    #         frame_sum_ot -= frame_diff
    #     frame_diff = frame_ixp1 - frame_ix
    #     frame_sum_ot += frame_diff

    # frame_diff = 0.4 * frame_ix + 0.6 * frame_diff
    # frame_diff = frame_diff / 255.0

    # cv2.imshow('Frame_diff {}'.format(fid), frame_img_data)
    # cv2.waitKey()

    # cv2.imshow('Frame_sum {}'.format(fid), frame_sum_ot)
    # cv2.waitKey()
    cv2.imwrite(os.path.join(img_writ_dir, game_run_frames[fid + 1]),
                frame_img_data)

    # if frame_diff is not None:
    #     cv2.imwrite(
    #         os.path.join(img_writ_dir[:-1] + '_diff',
    #                      game_run_frames[fid + 1]), frame_sum_ot)
    # cv2.imshow('Frame_diff {}'.format(fid), frame_diff)
    # cv2.waitKey()
    if wix == 1000:
        break

# Stitch images to video
imtypes_ = ['diff', 'direct']
imtype = imtypes_[1]
if imtype == 'diff':
    img_writ_dir = img_writ_dir[:-1] + '_diff'
frame_rate = 20
ffmpeg_arg_string = 'ffmpeg -y -framerate {} -i {}/KM_3486399_%d.png {}/out.mp4'.format(
    frame_rate, img_writ_dir, img_writ_dir)
ffmpeg_args = ffmpeg_arg_string.split(' ')
print(ffmpeg_args)
call(ffmpeg_args)
