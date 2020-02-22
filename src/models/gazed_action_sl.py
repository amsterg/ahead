import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load

np.random.seed(42)


class GAZED_ACTION_SL(nn.Module):
    def __init__(self, input_shape=(84, 84), load_model=False, epoch=0, num_actions=18):
        super(GAZED_ACTION_SL, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
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
        self.lin_in_shape = self.lin_in_shape()
        self.linear1 = nn.Linear(64*np.prod(self.lin_in_shape), 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.num_actions)
        self.batch_norm32 = nn.BatchNorm2d(32)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.load_model = load_model
        self.epoch = epoch

    def forward(self, x, x_g):
        # frame forward
        x = self.pool(self.relu(self.conv1(x)))
        x = self.batch_norm32(x)
        x = self.dropout(x)

        x = self.pool(self.relu(self.conv2(x)))
        x = self.batch_norm64(x)
        x = self.dropout(x)

        x = self.pool(self.relu(self.conv3(x)))
        x = self.batch_norm64(x)
        x = self.dropout(x)

        # gaze_overlay forward
        x_g = self.pool(self.relu(self.conv1(x_g)))
        x_g = self.batch_norm32(x_g)
        x_g = self.dropout(x_g)

        x_g = self.pool(self.relu(self.conv2(x_g)))
        x_g = self.batch_norm64(x_g)
        x_g = self.dropout(x_g)

        x_g = self.pool(self.relu(self.conv3(x_g)))
        x_g = self.batch_norm64(x_g)
        x_g = self.dropout(x_g)

        # combine gaze conv + frame conv
        x = 0.5*(x+x_g)
        x = x.view(-1, 64 * np.prod(self.lin_in_shape))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

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
        # TODO create as a wrapper
        # wrapper that gives num params

        # temp written down shape calcer
        out_shape = self.out_shape(self.conv1, self.input_shape)
        out_shape = self.out_shape(self.conv2, out_shape)
        out_shape = self.out_shape(self.conv3, out_shape)
        return out_shape

    def loss_fn(self, loss_, acts, targets):
        ce_loss = loss_(acts, targets)
        return ce_loss

    def train_loop(self, opt, lr_scheduler, loss_, x_var, y_var, xg_var=None, batch_size=32, gaze_pred=None):
        if xg_var is None:
            dataset = torch.utils.data.TensorDataset(x_var, y_var)
        else:
            dataset = torch.utils.data.TensorDataset(x_var, xg_var, y_var)

        train_data = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        self.val_data = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        if self.load_model:
            model_pickle = torch.load(
                self.model_save_string.format(self.epoch))
            self.load_state_dict(model_pickle['model_state_dict'])
            opt.load_state_dict(model_pickle['model_state_dict'])
            self.epoch = model_pickle['epoch']
            loss_val = model_pickle['loss']
        if gaze_pred is not None:
            assert xg_var is None
        for epoch in range(self.epoch, 20000):
            for i, data in enumerate(train_data):
                if xg_var is not None:
                    x, x_g, y = data
                else:
                    with torch.no_grad():
                        x, y = data
                        x_g = gaze_pred.infer(x)
                        x_g = x_g.unsqueeze(1).expand(x.shape)

                opt.zero_grad()

                acts = self.forward(
                    x, x_g)
                loss = self.loss_fn(loss_, acts, y)
                loss.backward()
                opt.step()

                if epoch % 10 == 0:
                    self.writer.add_histogram("acts", y)
                    self.writer.add_histogram("preds", acts)
                    self.writer.add_scalar('Loss', loss.data.item(), epoch)
                    self.writer.add_scalar(
                        'Acc', self.accuracy(xg_var, gaze_pred), epoch)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': loss,
                    }, self.model_save_string.format(epoch))

    def infer(self, x_var, xg_var):
        model_pickle = torch.load(
            self.model_save_string.format(self.epoch))
        self.load_state_dict(model_pickle['model_state_dict'])

        acts = self.forward(
            x_var, xg_var).argmax().data.numpy()

        return acts

    def accuracy(self, xg_var, gaze_pred):
        acc = 0
        ix = 0
        for i, data in enumerate(self.val_data):
            if xg_var is not None:
                x, x_g, y = data
            else:
                with torch.no_grad():
                    x, y = data
                    x_g = gaze_pred.infer(x)
                    x_g = x_g.unsqueeze(1).expand(x.shape)
            acts = self.forward(
                x, x_g).argmax(dim=1)
            acc += (acts == y).sum().item()
            ix += y.shape[0]
        return (acc/ix)


if __name__ == "__main__":

    action_net = GAZED_ACTION_SL()
    rand_frame = torch.rand(1, 4, 84, 84)
    rand_gaze = torch.rand(1, 4, 84, 84)
    # writer = SummaryWriter()

    action_net.writer.add_graph(
        action_net, input_to_model=(rand_frame, rand_gaze))
    action_net.writer.flush()
    # action_net.writer.close()
    rand_target = torch.randint(action_net.num_actions, [1])
    action_net.forward(rand_frame, rand_gaze)
    optimizer = torch.optim.Adadelta(
        action_net.parameters(), lr=1.0, rho=0.95)

    # if scheduler is declared, ought to use & update it , else model never trains
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)
    lr_scheduler = None

    loss_ = torch.nn.CrossEntropyLoss()
    action_net.train_loop(optimizer, lr_scheduler, loss_, rand_frame,
                          rand_target, rand_gaze, batch_size=4)
