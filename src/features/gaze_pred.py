from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from src.models.cnn_gaze import CNN_GAZE
from src.data.load_interim import load_gaze_data
from yaml import safe_load

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import cv2

transforms_ = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((80, 80)),
        transforms.Grayscale(),
        transforms.ToTensor(),

    ]
)


cnn_gaze_net = CNN_GAZE()
optimizer = torch.optim.Adam(cnn_gaze_net.parameters(), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda x: x*0.95)
loss_ = torch.nn.KLDivLoss(reduction='sum')

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

num_clusters = 20
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)


def gaze_clusters(gaze_data, num_clusters=num_clusters):
    if len(gaze_data) < num_clusters:
        dups = [gaze_data[-1] for _ in range(num_clusters-len(gaze_data))]
        gaze_data += dups
    kmeans.fit(gaze_data)
    return kmeans.cluster_centers_


def gaze_pdf(gaze):
    pdfs_true = []
    gaze_range = [76.0, 76.0]  # w,h
    # gaze_range = [160.0, 210.0]  # w,h

    x, y = np.mgrid[0:gaze_range[1]:1, 0:gaze_range[0]:1]

    pos = np.dstack((x, y))
    gpts = np.multiply(gaze, gaze_range).astype(np.int)
    for gpt in gpts:
        rv = multivariate_normal(mean=gpt[::-1], cov=2)
        pdfs_true.append(rv.pdf(pos))
    pdf = np.sum(pdfs_true, axis=0)
    wpdf = pdf/np.sum(pdf)
    # plt.imshow(pdf)
    # plt.imshow(wpdf)

    # plt.pause(12)
    assert abs(np.sum(wpdf)-1) <= 1e-2, print(np.sum(wpdf))

    return wpdf


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


images, gazes = load_gaze_data(stack=4)
assert len(images) == len(gazes)
img = np.array(images)
# print(img.shape)
# for im in img:
#     plt.imshow(im)
#     plt.pause(.3)

# image crop and oher transforms
images_ = [torch.stack([transforms_(np.clip(image_, 0, 1)).squeeze()
                        for image_ in image_stack]) for image_stack in images]
# print(np.array(images_).shape)

# transform gazes to tensors with clustering
# gazes_ = [torch.stack([torch.Tensor(gaze_clusters(gaze_, num_clusters))
#                        for gaze_ in gaze_stack]) for gaze_stack in gazes]

# transform gazes to tensors as a weighted distirbution
# gazes_ = [torch.stack([torch.Tensor(gaze_pdf(gaze_))
#                        for gaze_ in gaze_stack]) for gaze_stack in gazes]

# transform gazes to tensors as a weighted distirbution, and reduce stack
gazes_ = [reduce_gaze_stack(gaze_stack) for gaze_stack in gazes]

# print(np.array(gazes_).shape)

# exit()

# gazes_clustered = [gaze_clusters(gaze, num_clusters) for gaze in gazes]
# gazes_ = [torch.Tensor(gaze) for gaze in gazes_clustered]
assert len(images_) == len(gazes_)
x_variable = torch.stack(images_)
y_variable = torch.stack(gazes_)
# print(x_variable.shape)
# print(y_variable.shape)

test_ix = 0
image_ = images[test_ix]
gaze_ = gazes[test_ix]


def draw_figs(x_var, gazes):

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(x_var)
    fig.add_subplot(1, 2, 2)
    plt.imshow(gazes)
    plt.show()
    plt.waitforbuttonpress()
    plt.close('all')


def draw_clusters(clusters_):
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


# gaze = gazes[0]
# clusters = gaze_clusters(gaze_, num_clusters)
# draw_clusters(clusters)


# y_variable = torch.Tensor(clusters).unsqueeze(0)


batch_size = min(32, x_variable.shape[0])

# dataset normalizing
x_variable = (x_variable - torch.mean(x_variable)) * \
    torch.reciprocal(torch.std(x_variable))

draw_figs(x_var=image_[0], gazes=gazes_[test_ix].numpy())

# cnn_gaze_net.train_loop(optimizer, lr_scheduler, loss_, x_variable,
#                         y_variable, batch_size)


for cpt in tqdm(range(3000, 3020, 10)):
    smax = cnn_gaze_net.infer(cpt, x_variable[test_ix].unsqueeze(0))
    draw_figs(x_var=smax[0], gazes=gazes_[test_ix].numpy())
