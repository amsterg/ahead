from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from src.models.mdn import MDN, train_loop, infer
from src.models.cnn_gaze import CNN_GAZE
from src.data.data_loaders import load_gaze_data
from yaml import safe_load

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import cv2

transforms_ = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
])

num_gaussians = 10
num_clusters = 20
mdn = MDN(num_gaussians=num_gaussians)
optimizer = torch.optim.Adam(mdn.parameters(), lr=1e-5)

with open('src/config.yaml', 'r') as f:
    config_data = safe_load(f.read())

kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)


def gaze_clusters(gaze_data, num_clusters=num_gaussians):
    if len(gaze_data) < num_clusters:
        dups = [gaze_data[-1] for _ in range(num_clusters - len(gaze_data))]
        gaze_data += dups
    kmeans.fit(gaze_data)
    return kmeans.cluster_centers_


images, gazes = load_gaze_data()
assert len(images) == len(gazes)

# image = np.clip(image_, 0, 1)
# image = transforms_(image)
# # plt.imshow(image)
# # plt.pause(12)
# x_variable = image.unsqueeze(0)
# y_variable = torch.Tensor(gazes).unsqueeze(0)

images_ = [transforms_(np.clip(image_, 0, 1)) for image_ in images]
gazes_clustered = [gaze_clusters(gaze, num_clusters) for gaze in gazes]
gazes_ = [torch.Tensor(gaze) for gaze in gazes_clustered]
assert len(images_) == len(gazes_)
x_variable = torch.stack(images_)
y_variable = torch.stack(gazes_)

test_ix = 2
image_ = images[test_ix]
gaze_ = gazes[test_ix]
# plt.imshow(image_)
# plt.pause(2)


def draw_preds(model_cpt=0, x_infer=x_variable):
    pi_data, sig_data, mu_data = infer(mdn, model_cpt, x_infer)
    # print(pi_data, sig_data, mu_data)
    gaze_range = [160.0, 210.0]  # w,h

    mu_x = mu_data[0, :num_gaussians] * gaze_range[0]
    mu_y = mu_data[0, num_gaussians:] * gaze_range[1]

    sigma_x = sig_data[0, :num_gaussians] * gaze_range[0]
    sigma_y = sig_data[0, num_gaussians:] * gaze_range[1]

    x, y = np.mgrid[0:image_.shape[1]:1, 0:image_.shape[0]:1]
    print(image_.shape)
    pos = np.dstack((x, y))
    fig2 = plt.figure()
    fig3 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)
    pdfs_pred = []

    for m_x, m_y, s_x, s_y in zip(mu_x, mu_y, sigma_x, sigma_y):
        print(m_x, m_y, s_x, s_y)
        rv = multivariate_normal(mean=[m_x, m_y], cov=[s_x, s_y])
        # rv = multivariate_normal(mean=[m_x, m_y], cov=5)
        pdfs_pred.append(rv.pdf(pos))
    pdfs_true = []
    gpts = np.multiply(gaze_, gaze_range).astype(np.int)
    for gpt in gpts:
        rv = multivariate_normal(mean=gpt, cov=5)
        pdfs_true.append(rv.pdf(pos))

    wpdf_pred = np.sum(pdfs_pred, axis=0)
    print(wpdf_pred.shape)

    # print(wpdf.shape)
    # plt.imshow(wpdf_pred)
    # plt.pause(12)
    # exit()
    # ax2.contourf(x, y, wpdf_pred)
    y_lims = [gaze_range[1], 0]
    ax2.set_ylim(y_lims)
    ax2.set_title('preds_{}'.format(model_cpt))
    fig2.savefig('{}_{}.png'.format(config_data['VIS_PREDS_DIR'], model_cpt))
    wpdf_true = np.sum(pdfs_true, axis=0)
    # print(wpdf.shape)
    ax3.contourf(x, y, wpdf_true)
    # plt.ylim(plt.ylim()[::-1])
    ax3.set_ylim(y_lims)
    ax3.set_title('true_{}'.format(model_cpt))
    # fig3.savefig('{}_{}.png'.format(config_data['VIS_PREDS_DIR'], model_cpt))
    # fig3.close()
    # fig2.close()
    plt.show()

    # for pi, sig, mu in zip(pi_data, sig_data, mu_data):


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
# train_loop(mdn, optimizer, x_variable, y_variable, batch_size=32)

for cpt in tqdm(range(800, 820, 10)):
    draw_preds(model_cpt=cpt, x_infer=x_variable[test_ix].unsqueeze(0))
