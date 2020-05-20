import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoding_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(8, 8), stride=4),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1))

    def forward(self, x):
        out = self.encoding_net(x)
        embedding = out.flatten(start_dim=1).unsqueeze_(0)

        return out, embedding


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoding_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=1),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2),
            nn.ConvTranspose2d(32, 1, kernel_size=(8, 8), stride=4))
        self.softmax = torch.nn.Softmax2d()

    def forward(self, x):
        out = self.softmax(self.decoding_net(x))
        return out


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()
        self.gate = nn.GRU(3136, 1, 1)

    def forward(self, x):
        _, out = self.gate(x, torch.ones(1, 1, 1) * -1)
        gate_output = torch.relu(torch.sign(out))
        return gate_output


class SEA(nn.Module):
    def __init__(self, OUTPUT):
        super(SEA, self).__init__()
        self.gaze_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 8), stride=4),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1))
        self.fully_connected = nn.Sequential(torch.nn.Linear(6272, 512),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(512, OUTPUT))
        self.encode = Encoder()
        self.decode = Decoder()
        self.gate = Gate()

    def forward(self, x):
        # Eye gaze network
        encoded, frame_embedding = self.encode(x)
        decoded = self.decode(encoded)

        # Gating network
        gate_output = self.gate(frame_embedding)
        weighted_gaze = gate_output * decoded

        # Action network
        gaze_conv = self.gaze_conv(weighted_gaze)
        gaze_embedding = gaze_conv.flatten(start_dim=1).unsqueeze_(0)
        concat_out = torch.cat((frame_embedding, gaze_embedding), dim=-1)
        out = self.fully_connected(concat_out)
        out = torch.log_softmax(out, dim=1)
        return out, decoded

    def train_loop(self, optimizer, frame, gaze, action_target):
        gaze_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        action_criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(2):
            optimizer.zero_grad()
            action_out, predicted_gaze = self.forward(frame)

            action_loss = action_criterion(action_out, action_target.long())
            gaze_loss = gaze_criterion(predicted_gaze, gaze)
            loss = gaze_loss + action_loss
            loss.backward()
            optimizer.step()


if __name__ == "__main__":

    # Selective Eye gaze Augmentation Network
    sea = SEA(4)
    rand_image = torch.rand(1, 4, 84, 84)
    rand_gaze = torch.rand(1, 1, 84, 84)
    rand_target = torch.rand(1, 4)

    # Optimizer
    optimizer = torch.optim.Adadelta(sea.parameters(), lr=1.0, rho=0.95)

    # Training loop
    sea.train_loop(optimizer, rand_image, rand_gaze, rand_target)
