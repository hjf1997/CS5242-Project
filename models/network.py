import torch
import torch.nn as nn
from torchvision import models
from convlstm import ConvLSTM
import torch.nn.functional as F

class Lstm(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.parameters():
            param.requires_grad = False

        self.enc_1 = nn.Sequential(*vgg16.features)
        self.conv_lstm = ConvLSTM(512, 128, (3, 3), 2, batch_first=True, return_all_layers=True)
        self.conv_1 = nn.Conv2d(128, 128, (8, 14))
        self.conv_2 = nn.Conv2d(128, 128, (8, 14))
        self.de_cell1 = nn.LSTMCell(128, 128)
        self.de_cell2 = nn.LSTMCell(128, 128)
        self.fc_os = nn.Linear(128, 35)
        self.fc_r = nn.Linear(128, 82)
        self.attention = torch.nn.MultiheadAttention(128, 1)

    def forward(self, images):
        b, t, _, _, _ = images.shape
        images = images.reshape((-1, *images.shape[2:]))
        images = self.enc_1(images)
        images = images.reshape((b, t, *images.shape[1:]))
        layer_output, last_states = self.conv_lstm(images)
        layer_output = layer_output[1]
        for i in range(2):
            last_states[i][0], last_states[i][1] = \
                torch.relu(self.conv_1(last_states[i][0]).squeeze(-1).squeeze(-1)), \
                torch.relu(self.conv_2(last_states[i][1]).squeeze(-1).squeeze(-1))
        layer_output = layer_output.reshape((-1, *layer_output.shape[2:]))
        keys = self.conv_1(layer_output).reshape((b, t, -1)).transpose(0, 1)
        # decoder
        x = torch.zeros((images.shape[0], 128), device=images.device)
        states = []
        for i in range(3):
            if i == 0:
                h_1, c_1 = self.de_cell1(x, (last_states[0][0], last_states[0][1]))
                h_2, c_2 = self.de_cell2(h_1, (last_states[1][0], last_states[1][1]))
            else:
                h_1, c_1 = self.de_cell1(h_2, (h_1, c_1))
                h_2, c_2 = self.de_cell2(h_1, (h_2, c_2))
            h_2, _ = self.attention(h_2.unsqueeze(0), keys, keys)
            h_2 = h_2.squeeze(0)
            states.append(h_2)
        object = self.fc_os(states[0])
        relation = self.fc_r(states[1])
        subject = self.fc_os(states[2])

        return object, relation, subject


class Vgg(nn.Module):

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.parameters():
            param.requires_grad = False
        self.enc_1 = nn.Sequential(*vgg16.features)

        self.avg_pool = nn.AdaptiveAvgPool2d((7,7))
        self.fc1 = nn.Linear(25088, 1000)
        self.fc2 = nn.Linear(1000, 35)
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()

    def forward(self, images):

        x = self.enc_1(images)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
