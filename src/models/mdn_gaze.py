from random import shuffle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
from math import floor
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load

np.random.seed(42)


class MDN(nn.Module):
    def __init__(self, input_shape=(80, 80), num_gaussians=10, load_model=False, epoch=200):
        super(MDN, self).__init__()
        self.input_shape = input_shape
        self.num_gaussians = num_gaussians
        with open('src/config.yaml', 'r') as f:
            config_data = safe_load(f.read())
        self.config_yml = config_data
        self.writer = SummaryWriter()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), (1, 1))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 20, 5)
        self.lin_in_shape = self.lin_in_shape()
        self.fc1 = nn.Linear(20 * np.prod(self.lin_in_shape), num_gaussians*12)
        self.fc2 = nn.Linear(num_gaussians*12, num_gaussians*9)
        self.fc3 = nn.Linear(num_gaussians*9, num_gaussians*6)
        self.z_pi = nn.Linear(num_gaussians*6, num_gaussians)
        self.z_sigma = nn.Linear(num_gaussians*6, num_gaussians*2)
        self.z_mu = nn.Linear(num_gaussians*6, num_gaussians*2)
        self.softplus = torch.nn.Softplus()
        # self.z_rho = nn.Linear(num_gaussians*6, num_gaussians)
        if load_model:
            model_pickle = torch.load(
                self.config_yml['MODEL_SAVE_DIR']+'Epoch_{}.pt'.format(epoch))
            self.load_state_dict(model_pickle['model_state_dict'])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 20 * np.prod(self.lin_in_shape))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = F.relu(self.fc3(x))
        pi = nn.functional.softmax(self.z_pi(z), -1)
        # sigma = torch.exp(self.z_sigma(z))
        sigma = self.softplus(self.z_sigma(z))
        mu = self.z_mu(z)
        # rho = self.z_rho(z)*0

        # return pi, sigma, mu, rho
        mu = torch.clamp(mu, 0.1, 0.9)
        sigma = torch.clamp(sigma, 0.1, 0.3)

        return pi, sigma, mu

    def out_shape(self, layer, in_shape):
        h_in, w_in = in_shape
        h_out, w_out = floor(
            ((h_in + 2 * layer.padding[0] - layer.dilation[0] *
              (layer.kernel_size[0] - 1) - 1) / layer.stride[0]) + 1

        ), floor(
            ((w_in + 2 * layer.padding[1] - layer.dilation[1] *
              (layer.kernel_size[1] - 1) - 1) / layer.stride[1]) + 1

        )
        return h_out, w_out

    def lin_in_shape(self):
        outs = self.out_shape(self.conv1, self.input_shape)
        outs = self.out_shape(self.pool, outs)
        outs = self.out_shape(self.conv2, outs)
        outs = self.out_shape(self.pool, outs)
        outs = self.out_shape(self.conv3, outs)
        outs = self.out_shape(self.pool, outs)
        return outs

    def gaussian_distribution(self, y, mu, sigma, rho=0):
        # mu_x = mu[:, :self.num_gaussians]
        mu_x = mu[:self.num_gaussians]
        # mu_y = mu[:, self.num_gaussians:]
        mu_y = mu[self.num_gaussians:]
        assert mu_x.shape == mu_y.shape, print(mu_x.shape, mu_y.shape)

        # sigma_x = sigma[:, :self.num_gaussians]
        sigma_x = sigma[:self.num_gaussians]
        # sigma_y = sigma[:, self.num_gaussians:]
        sigma_y = sigma[self.num_gaussians:]
        # mu_x = torch.clamp(mu_x, 0.1, 0.9)
        # mu_y = torch.clamp(mu_y, 0.1, 0.9)
        # sigma_x = torch.clamp(sigma_x, 0.1, 0.3)
        # sigma_y = torch.clamp(sigma_y, 0.1, 0.3)
        assert sigma_x.shape == sigma_y.shape, print(
            sigma_x.shape, sigma_y.shape)
        self.writer.add_histogram('mu_x', mu_x)
        self.writer.add_histogram('mu_y', mu_y)
        self.writer.add_histogram('sigma_x', sigma_x)
        self.writer.add_histogram('sigma_y', sigma_y)

        y = y.squeeze()
        # coord = y[torch.randint(0, len(y), (1, 1)).item()]
        result = 0

        for coord in y:
            cx = coord[0].expand(mu_x.shape[0]).reshape(1, -1)
            cy = coord[1].expand(mu_y.shape[0]).reshape(1, -1)
            x_val = (torch.pow(cx-mu_x, 2) *
                     torch.reciprocal(torch.pow(sigma_x, 2)))
            y_val = (torch.pow(cy-mu_y, 2) *
                     torch.reciprocal(torch.pow(sigma_y, 2)))
            num = torch.exp(-(x_val + y_val))
            # self.writer.add_histogram('x_val', x_val)
            # self.writer.add_histogram('y_val', y_val)

            pdf = num * \
                torch.reciprocal(2*np.pi*torch.mul(sigma_x, sigma_y))

            # x_val = (torch.pow(cx-mu_x, 2) *
            #          torch.reciprocal(torch.pow(sigma_x, 2)))
            # y_val = (torch.pow(cy-mu_y, 2) *
            #          torch.reciprocal(torch.pow(sigma_y, 2)))
            # xy_val = (2*rho*(cx-mu_x)*(cy-mu_y) *
            #           torch.reciprocal(torch.mul(sigma_x, sigma_y)))
            # num = torch.exp(-torch.reciprocal(2*(1-torch.pow(rho, 2)))
            #                 * (x_val + y_val - xy_val))
            # self.writer.add_histogram('exp', num)

            # pdf = num * \
            #     torch.reciprocal(2*np.pi*torch.mul(sigma_x, sigma_y)
            #                      * torch.sqrt(1-torch.pow(rho, 2)))

            result += pdf
        result /= len(y)

        return result

    def sos(self, y, mu, sigma):
        # mu_x = mu[:, :self.num_gaussians]
        mu_x = mu[:self.num_gaussians]
        # mu_y = mu[:, self.num_gaussians:]
        mu_y = mu[self.num_gaussians:]
        assert mu_x.shape == mu_y.shape, print(mu_x.shape, mu_y.shape)

        # sigma_x = sigma[:, :self.num_gaussians]
        sigma_x = sigma[:self.num_gaussians]
        # sigma_y = sigma[:, self.num_gaussians:]
        sigma_y = sigma[self.num_gaussians:]
        assert sigma_x.shape == sigma_y.shape, print(
            sigma_x.shape, sigma_y.shape)

        y = y.squeeze()
        # coord = y[torch.randint(0, len(y), (1, 1)).item()]
        result = 0

        for coord in y:
            cx = coord[0].expand(mu_x.shape[0]).reshape(1, -1)
            cy = coord[1].expand(mu_y.shape[0]).reshape(1, -1)
            x_val = torch.pow(cx-mu_x, 2)
            y_val = torch.pow(cy-mu_y, 2)
            pdf = 0
            result += pdf
        result /= len(y)

        return result

    def loss_fn(self, pi_b, sigma_b, mu_b, y_b):
        result = []
        for pi, sig, mu, y in zip(pi_b, sigma_b, mu_b, y_b):
            res = self.gaussian_distribution(y, mu, sig) * pi
            res = torch.sum(res, dim=1)
            res = -torch.log(res)
            result.append(res)
        result = torch.mean(torch.stack(result))
        # result = torch.pow(result, 2)
        return result

    def loss_fn_sos(self, pi_b, sigma_b, mu_b, y_b):
        result = []
        for pi, sig, mu, y in zip(pi_b, sigma_b, mu_b, y_b):
            res = self.sos(y, mu, sig)
            res = torch.sum(res, dim=1)
            res = -torch.log(res)
            result.append(res)
        result = torch.mean(torch.stack(result))
        # result = torch.pow(result, 2)
        return result


def train_loop(net, opt, x_var, y_var, batch_size):
    # if x_var.shape[0] > batch_size:
    #     print(x_var.shape)
    #     print(y_var.shape)
    #     exit()
    for epoch in range(10000):

        pi_variable, sigma_variable, mu_variable = net(
            x_var)

        loss = net.loss_fn(pi_variable, sigma_variable,
                           mu_variable, y_var)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

        if epoch % 10 == 0:
            net.writer.add_scalar('Loss', loss.data.item(), epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss,
            }, net.config_yml['MODEL_SAVE_DIR']+'Epoch_{}.pt'.format(epoch))


def infer(net, epoch, x_var):
    model_pickle = torch.load(
        net.config_yml['MODEL_SAVE_DIR']+'Epoch_{}.pt'.format(epoch))
    net.load_state_dict(model_pickle['model_state_dict'])

    pi, sig, mu = net(x_var)

    pi_data = pi.data.numpy()
    sig_data = sig.data.numpy()
    mu_data = mu.data.numpy()
    return pi_data, sig_data, mu_data


def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)


def batch_splitter(x, y):
    assert x.shape[0] == y.shape[0]
    ixs = list(range(x.shape[0]))

    print(x)


if __name__ == "__main__":

    rand_image = np.random.random((80, 80, 1))

    # cv2.imshow('rand_image', rand_image)
    # cv2.waitKey()
    rand_image = rand_image.reshape(1, 80, 80)
    # rand_y = [[np.random.randint(0, 80), np.random.randint(0, 80)]
    #           for _ in range(42)]
    rand_y = [[np.random.random(), np.random.random()]
              for _ in range(42)]

    x_variable = torch.autograd.Variable(torch.Tensor(rand_image).unsqueeze(0))
    y_variable = torch.autograd.Variable(torch.Tensor(rand_y).unsqueeze(0))

    batch_splitter(x_variable, y_variable)

    exit()
    mdn = MDN()

    optimizer = torch.optim.Adam(mdn.parameters(), lr=1e-3)
    # train_loop(mdn, optimizer, x_variable, y_variable)
    infer(mdn, 10, x_variable)
