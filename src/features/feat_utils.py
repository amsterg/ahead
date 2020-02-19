import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from torchvision import transforms
import torch

NUM_CLUSTERS = 20
kmeans = KMeans(init='k-means++', n_clusters=NUM_CLUSTERS, n_init=10)


def draw_figs(x_var, x_title=0, gazes=None):
    fig = plt.figure()
    fig.suptitle(x_title)
    fig.add_subplot(1, 2, 1)
    plt.imshow(x_var, cmap='RdPu')
    fig.add_subplot(1, 2, 2)
    plt.imshow(gazes, cmap='RdPu')
    plt.show()
    plt.waitforbuttonpress()
    plt.close('all')


def gaze_clusters(gaze_data, num_clusters=NUM_CLUSTERS):
    if len(gaze_data) < num_clusters:
        dups = [gaze_data[-1] for _ in range(num_clusters-len(gaze_data))]
        gaze_data += dups
    kmeans.fit(gaze_data)
    return kmeans.cluster_centers_


def gaze_pdf(gaze):
    pdfs_true = []
    gaze_range = [84, 84]  # w,h
    # gaze_range = [160.0, 210.0]  # w,h

    gaze_map = wpdf = np.zeros(gaze_range)

    gpts = np.multiply(gaze, gaze_range).astype(np.int)
    gpts = np.clip(gpts, 0, 83).astype(np.int)

    x, y = np.mgrid[0:gaze_range[1]:1, 0:gaze_range[0]:1]
    pos = np.dstack((x, y))
    for gpt in gpts:
        rv = multivariate_normal(
            mean=gpt[::-1], cov=[[2.85*2.85, 0], [0, 2.92*2.92]])
        pdfs_true.append(rv.pdf(pos))
    pdf = np.sum(pdfs_true, axis=0)
    wpdf = pdf/np.sum(pdf)
    gaze_map = wpdf
    assert abs(np.sum(wpdf)-1) <= 1e-2, print(np.sum(wpdf))

    # for gpt in gpts:
    #     gaze_map[gpt[1], gpt[0]] = 1
    # gaze_map = gaze_map/np.sum(gaze_map)

    # draw_figs(wpdf, gaze_map)

    assert abs(np.sum(gaze_map)-1) <= 1e-2, print(np.sum(gaze_map))

    return gaze_map


def reduce_gaze_stack(gaze_stack):
    gaze_pdfs = [gaze_pdf(gaze) for gaze in gaze_stack]
    pdf = np.sum(gaze_pdfs, axis=0)
    wpdf = pdf/np.sum(pdf)
    # print(torch.Tensor(wpdf).shape)
    # plt.imshow(wpdf)
    # plt.pause(12)
    # exit()
    assert abs(np.sum(wpdf)-1) <= 1e-2, print(np.sum(wpdf))

    return torch.Tensor(wpdf)


def image_transforms(image_size=(84, 84)):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.Grayscale(),
            transforms.ToTensor(),

        ]
    )


def draw_clusters(clusters_, image_, gaze_):
    x, y = np.mgrid[0:image_.shape[1]:1, 0:image_.shape[0]:1]

    pos = np.dstack((x, y))
    fig2 = plt.figure()
    fig3 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)
    gaze_range = [160.0, 210.0]  # w,h

    pdfs_clus = []
    gpts = np.multiply(clusters_, gaze_range).astype(np.int)
    for gpt in gpts:
        rv = multivariate_normal(mean=gpt, cov=5)
        pdfs_clus.append(rv.pdf(pos))

    pdfs_true = []
    gpts = np.multiply(gaze_, gaze_range).astype(np.int)
    for gpt in gpts:
        rv = multivariate_normal(mean=gpt, cov=5)
        pdfs_true.append(rv.pdf(pos))

    wpdf_clus = np.sum(pdfs_clus, axis=0)
    # print(wpdf_clus.shape)
    ax2.contourf(x, y, wpdf_clus)
    y_lims = [gaze_range[0], 0]
    ax2.set_ylim(y_lims)

    wpdf_true = np.sum(pdfs_true, axis=0)
    # print(wpdf_true.shape)
    ax3.contourf(x, y, wpdf_true)
    # plt.ylim(plt.ylim()[::-1])
    ax3.set_ylim(y_lims)

    plt.show()
