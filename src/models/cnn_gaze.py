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


class CNN_GAZE(nn.Module):
    def __init__(self, input_shape=(80, 80), load_model=False, epoch=200):
        super(CNN_GAZE, self).__init__()
        self.input_shape = input_shape
        with open('src/config.yaml', 'r') as f:
            config_data = safe_load(f.read())
        self.config_yml = config_data
        self.writer = SummaryWriter()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=(4, 4))
        # self.pool = nn.MaxPool2d((1, 1), (1, 1), (0, 0), (1, 1))
        self.pool = lambda x: x
        self.conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=(2, 2))
        self.deconv3 = nn.ConvTranspose2d(32, 1, 8, stride=(4, 4))
        self.batch_norm32 = nn.BatchNorm2d(32)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout()

        # self.lin_in_shape = self.lin_in_shape()
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax2d()
        if load_model:
            model_pickle = torch.load(
                self.config_yml['model_save_dir']+'Epoch_{}.pt'.format(epoch))
            self.load_state_dict(model_pickle['model_state_dict'])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm32(x)
        x = self.dropout(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch_norm64(x)
        x = self.dropout(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.batch_norm64(x)
        x = self.dropout(x)

        x = self.pool(F.relu(self.deconv1(x)))
        x = self.batch_norm64(x)
        x = self.dropout(x)

        x = self.pool(F.relu(self.deconv2(x)))
        x = self.batch_norm32(x)
        x = self.dropout(x)

        x = self.pool(F.relu(self.deconv3(x)))
        # x = F.batch_norm(x)
        # x = self.dropout(x)
        # print(x.shape)

        x = x.squeeze(1)
        # print(x.shape)
        x = F.softmax(x, dim=1)
        x = F.softmax(x, dim=2)
        x = torch.clamp(x, min=-1, max=1)

        return x

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

    # def lin_in_shape(self):
    #     outs = self.out_shape(self.conv1, self.input_shape)
    #     outs = self.out_shape(self.pool, outs)
    #     outs = self.out_shape(self.conv2, outs)
    #     outs = self.out_shape(self.pool, outs)
    #     outs = self.out_shape(self.conv3, outs)
    #     outs = self.out_shape(self.pool, outs)
    #     outs = self.out_shape(self.deconv1, outs)
    #     outs = self.out_shape(self.pool, outs)
    #     outs = self.out_shape(self.deconv2, outs)
    #     outs = self.out_shape(self.pool, outs)
    #     outs = self.out_shape(self.deconv3, outs)
    #     outs = self.out_shape(self.pool, outs)

    #     return outs

    def loss_fn(self, loss_, smax_pi, targets):
        # print(smax_pi.shape)
        # print(targets.shape)
        kl_loss = loss_(smax_pi, targets)
        # kl_loss_sum = torch.sum(kl_loss, dim=[0, 1, 2])
        # print(kl_loss.shape)
        # print(kl_loss_sum.shape)
        # exit()
        return kl_loss

    def train_loop(self, opt, lr_scheduler, loss_, x_var, y_var, batch_size=32):
        # if x_var.shape[0] > batch_size:
        #     print(x_var.shape)
        #     print(y_var.shape)
        #     # exit()
        dataset = torch.utils.data.TensorDataset(x_var, y_var)
        train_data = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(20000):
            for i, data in enumerate(train_data):
                x, y = data
                smax_pi = self.forward(
                    x)

                # smax_pi_lps = torch.log(smax_pi.squeeze())
                # plt.imshow(smax_pi.data.numpy()[0][0])
                # plt.waitforbuttonpress()
                loss = self.loss_fn(loss_, smax_pi.log(), y)
                # loss = torch.nn.functional.mse_loss(smax_pi, y)
                opt.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                opt.step()
                # if epoch % 500 == 0:
                #     lr_scheduler.step()

                if epoch % 10 == 0:
                    self.writer.add_histogram('smax', smax_pi[0])
                    self.writer.add_histogram('target', y)
                    self.writer.add_scalar('Loss', loss.data.item(), epoch)
                    fig = plt.figure()
                    fig.add_subplot(1, 2, 1)
                    plt.imshow(smax_pi.data.numpy()[0])
                    fig.add_subplot(1, 2, 2)
                    plt.imshow(y[0])
                    plt.show()
                    plt.waitforbuttonpress()
                    plt.close('all')

                # if epoch % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': loss,
                    }, self.config_yml['model_save_dir']+'Epoch_{}.pt'.format(epoch))

    def infer(self, epoch, x_var):
        model_pickle = torch.load(
            self.config_yml['model_save_dir']+'Epoch_{}.pt'.format(epoch))
        self.load_state_dict(model_pickle['model_state_dict'])

        smax_dist = self.forward(x_var).data.numpy()

        return smax_dist


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
