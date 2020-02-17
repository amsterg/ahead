import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load

np.random.seed(42)


class CNN_GAZE(nn.Module):
    def __init__(self, input_shape=(84, 84), load_model=False, epoch=0):
        super(CNN_GAZE, self).__init__()
        self.input_shape = input_shape

        with open('src/config.yaml', 'r') as f:
            self.config_yml = safe_load(f.read())
        self.model_save_string = self.config_yml['MODEL_SAVE_DIR']+"{}".format(
            self.__class__.__name__)+'_Epoch_{}.pt'

        self.writer = SummaryWriter()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=(4, 4))
        self.pool = nn.MaxPool2d((1, 1), (1, 1), (0, 0), (1, 1))
        # self.pool = lambda x: x

        self.conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=(2, 2))
        self.deconv3 = nn.ConvTranspose2d(32, 1, 8, stride=(4, 4))
        self.batch_norm32 = nn.BatchNorm2d(32)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout()

        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax2d()
        self.load_model = load_model
        self.epoch = epoch

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

        x = self.deconv3(x)

        x = x.squeeze(1)

        x = x.view(-1, x.shape[1]*x.shape[2])

        x = F.log_softmax(x, dim=1)

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

    def lin_in_shape(self):
        # TODO
        # wrapper that gives num params
        pass

    def loss_fn(self, loss_, smax_pi, targets):

        targets_reshpaed = targets.view(-1, targets.shape[1]*targets.shape[2])

        kl_loss = loss_(smax_pi, targets_reshpaed)

        return kl_loss

    def train_loop(self, opt, lr_scheduler, loss_, x_var, y_var, batch_size=32):
        dataset = torch.utils.data.TensorDataset(x_var, y_var)
        train_data = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        if self.load_model:
            model_pickle = torch.load(
                self.model_save_string.format(self.epoch))
            self.load_state_dict(model_pickle['model_state_dict'])
            opt.load_state_dict(model_pickle['model_state_dict'])
            self.epoch = model_pickle['epoch']
            loss_val = model_pickle['loss']

        for epoch in range(self.epoch, 20000):
            for i, data in enumerate(train_data):
                x, y = data

                opt.zero_grad()

                smax_pi = self.forward(
                    x)

                loss = self.loss_fn(loss_, smax_pi, y)
                loss.backward()
                opt.step()

                if epoch % 100 == 0:
                    self.writer.add_histogram('smax', smax_pi[0])
                    self.writer.add_histogram('target', y)
                    self.writer.add_scalar('Loss', loss.data.item(), epoch)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': loss,
                    }, self.model_save_string.format(epoch))

    def infer(self, epoch, x_var):
        model_pickle = torch.load(
            self.model_save_string.format(epoch))
        self.load_state_dict(model_pickle['model_state_dict'])

        smax_dist = self.forward(
            x_var).view(-1, self.input_shape[0], self.input_shape[1]).data.numpy()

        return smax_dist


if __name__ == "__main__":

    rand_image = torch.rand(4, 84, 84)
    rand_target = torch.rand(4, 84, 84)

    cnn_gaze_net = CNN_GAZE()
    cnn_gaze_net.lin_in_shape()
    optimizer = torch.optim.Adadelta(
        cnn_gaze_net.parameters(), lr=1.0, rho=0.95)

    # if scheduler is declared, ought to use & update it , else model never trains
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)
    lr_scheduler = None

    loss_ = torch.nn.KLDivLoss(reduction='batchmean')
    cnn_gaze_net.train_loop(optimizer, lr_scheduler, loss_, rand_image,
                            rand_target, batch_size=4)
