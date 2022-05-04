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
        dups = [gaze_data[-1] for _ in range(num_clusters - len(gaze_data))]
        gaze_data += dups
    kmeans.fit(gaze_data)
    return kmeans.cluster_centers_


def gaze_pdf(gaze, gaze_count=1):
    pdfs_true = []
    gaze_range = [84, 84]  # w,h
    # gaze_range = [160.0, 210.0]  # w,h

    gaze_map = wpdf = np.zeros(gaze_range)

    gpts = np.multiply(gaze, gaze_range).astype(np.int)
    gpts = np.clip(gpts, 0, 83).astype(np.int)

    x, y = np.mgrid[0:gaze_range[1]:1, 0:gaze_range[0]:1]
    pos = np.dstack((x, y))
    if gaze_count != -1:
        gpts = gpts[-gaze_count:]

    for gpt in gpts:
        rv = multivariate_normal(mean=gpt[::-1],
                                 cov=[[2.85 * 2.85, 0], [0, 2.92 * 2.92]])
        pdfs_true.append(rv.pdf(pos))
    pdf = np.sum(pdfs_true, axis=0)
    wpdf = pdf / np.sum(pdf)
    gaze_map = wpdf
    # assert abs(np.sum(wpdf) - 1) <= 1e-2, print(np.sum(wpdf))

    # for gpt in gpts:
    #     gaze_map[gpt[1], gpt[0]] = 1
    # gaze_map = gaze_map/np.sum(gaze_map)
    # draw_figs(wpdf, gazes=gaze_map)
    assert abs(np.sum(gaze_map) - 1) <= 1e-2, print(np.sum(gaze_map))

    return gaze_map


def reduce_gaze_stack(gaze_stack):
    gaze_pdfs = [gaze_pdf(gaze) for gaze in gaze_stack]
    pdf = np.sum(gaze_pdfs, axis=0)
    wpdf = pdf / np.sum(pdf)
    # print(torch.Tensor(wpdf).shape)
    # plt.imshow(wpdf)
    # plt.pause(12)
    # exit()
    assert abs(np.sum(wpdf) - 1) <= 1e-2, print(np.sum(wpdf))

    return torch.Tensor(wpdf)


def fuse_gazes(images_, gazes, gaze_count=-1):
    gazes_ = [
        torch.stack([
            torch.Tensor(gaze_pdf(gaze_, gaze_count)) for gaze_ in gaze_stack
        ]) for gaze_stack in gazes
    ]
    fused = images_ * torch.stack(gazes_)
    # print(fused.shape)
    # for img in images_[0]:
    #     plt.imshow(img)
    #     plt.show()
    # for img in gazes_[0]:
    #     plt.imshow(img)
    #     plt.show()
    # for img in fused[0]:
    #     plt.imshow(img)
    #     plt.show()
    # exit()
    return fused


def fuse_gazes_np(images_, gazes, gaze_count=-1):
    gazes_ = [
        np.stack([gaze_pdf(gaze_, gaze_count) for gaze_ in gaze_stack])
        for gaze_stack in gazes
    ]
    fused = np.stack(images_) * np.stack(gazes_)

    return np.stack(gazes_)


def fuse_gazes_noop(images_,
                    gazes_,
                    actions,
                    gaze_count=-1,
                    fuse_type='stack',
                    fuse_val=1):
    actions_tensor = torch.sign(torch.LongTensor(actions))

    if fuse_type == 'stack':

        assert len(actions_tensor.shape) == 2, print(
            "Wrong data type for action for chosen fuse type")

        action_mask = torch.stack([
            torch.stack([
                torch.ones((84, 84)) if a == 1 else torch.zeros((84, 84))
                for a in ast
            ]) for ast in actions_tensor
        ])

    elif fuse_type == 'last':

        assert len(actions_tensor.shape) == 1, print(
            "Wrong data type for action for chosen fuse type")

        action_mask = torch.stack([
            torch.ones((4, 84, 84)) if a == 1 else torch.zeros((4, 84, 84))
            for a in actions_tensor
        ])

    gazes_ = [
        torch.stack([
            torch.Tensor(gaze_pdf(gaze_, gaze_count)) for gaze_ in gaze_stack
        ]) for gaze_stack in gazes_
    ]

    gazes_ = action_mask.to(device=torch.device('cuda')) * torch.stack(
        gazes_).to(device=torch.device('cuda'))

    if fuse_val == 1:
        gazes_ = torch.stack([
            torch.stack([
                torch.ones((84, 84)).to(
                    device=torch.device('cuda')) if torch.sum(g) == 0 else g
                for g in gst
            ]) for gst in gazes_
        ])

    elif fuse_val == 0:
        gazes_ = torch.stack([
            torch.stack([
                torch.zeros((84, 84)).to(
                    device=torch.device('cuda')) if torch.sum(g) == 0 else g for g in gst
            ]) for gst in gazes_
        ])
    else:
        raise Exception("Improper fuse val")

    fused = images_.to(device=torch.device('cuda')) * gazes_.to(
        device=torch.device('cuda'))

    # print(torch.stack(images_).shape)
    # print(actions_tensor.shape)
    # print(np.array(gazes).shape)
    # print(action_mask.shape)
    # print(actions)
    # for img in images_[2]:
    #     plt.imshow(img)
    #     plt.show()
    # for img in gazes_[2]:
    #     print(torch.sum(img))

    #     plt.imshow(img)
    #     plt.show()
    # for img in fused[2]:
    #     plt.imshow(img)
    #     plt.show()
    # exit()

    return fused.to(device='cpu')


def normalize(img, val):
    return (img - img.min()) / (img.max() - img.min()) * val


def image_transforms(image_size=(84, 84), to_tensor=True):
    transforms_ = [
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.Grayscale()
    ]

    if to_tensor:
        transforms_.append(transforms.ToTensor())

    return transforms.Compose(transforms_)


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


def transform_images(images, type='torch'):
    if type == 'torch':

        transforms_ = image_transforms()
        images = torch.stack([
            torch.stack(
                [transforms_(image_).squeeze() for image_ in image_stack])
            for image_stack in images
        ])

    elif type == 'numpy':

        transforms_ = image_transforms(to_tensor=False)
        images = [
            np.stack([transforms_(image_) for image_ in image_stack])
            for image_stack in images
        ]

    return images