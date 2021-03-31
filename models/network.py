import torch
import torch.nn as nn
from torchvision import models
from convlstm import ConvLSTM
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class Lstm(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_1 = Vgg()
        self.conv_lstm = ConvLSTM(128, 128, (3, 3), 1, batch_first=True, return_all_layers=True)
        self.conv_1 = nn.Conv2d(128, 128, (7,7))
        self.conv_2 = nn.Conv2d(128, 128, (7,7))
        self.de_cell1 = nn.LSTMCell(128, 128)
        # self.de_cell2 = nn.LSTMCell(128, 128)
        self.fc_in_lstm = nn.Linear(35, 128)
        self.fc_os = nn.Linear(128, 35)
        self.fc_r = nn.Linear(128, 82)
        self.attention = torch.nn.MultiheadAttention(128, 1)

    def forward(self, images):
        b, t, _, _, _ = images.shape
        images = images.reshape((-1, *images.shape[2:]))
        images, pred_label = self.enc_1(images)
        images = images.reshape((b, t, *images.shape[1:]))
        pred_label = pred_label.reshape((b, t, -1))
        layer_output, last_states = self.conv_lstm(images)
        layer_output = layer_output[0]
        for i in range(1):
            last_states[i][0], last_states[i][1] = \
                torch.relu(self.conv_1(last_states[i][0]).squeeze(-1).squeeze(-1)), \
                torch.relu(self.conv_2(last_states[i][1]).squeeze(-1).squeeze(-1))
        layer_output = layer_output.reshape((-1, *layer_output.shape[2:]))
        keys = self.conv_1(layer_output).reshape((b, t, -1)).transpose(0, 1)
        # decoder
        pred_label = torch.mean(pred_label, dim=1)
        states = []
        for i in range(3):
            if i == 0:
                h_1, c_1 = self.de_cell1(self.fc_in_lstm(pred_label), (last_states[0][0], last_states[0][1]))
            else:
                h_1, c_1 = self.de_cell1(h_2, (h_1, c_1))
            h_2, _ = self.attention(h_1.unsqueeze(0), keys, keys)
            h_2 = h_2.squeeze(0)
            states.append(h_2)
        object = self.fc_os(states[0])
        relation = self.fc_r(states[1])
        subject = self.fc_os(states[2])

        return pred_label, object, relation, subject


class VggCla(nn.Module):

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

class Vgg(nn.Module):

    def __init__(self):
        super().__init__()
        efficient = EfficientNet.from_pretrained("efficientnet-b2", advprop=True)
        for param in efficient.parameters():
            param.requires_grad = False
        self.efficient = efficient

        self.conv1 = nn.Conv2d(1408, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(6272, 1000)
        self.fc2 = nn.Linear(1000, 35)
        self.dropout = nn.Dropout(0.6)
        self.flatten = nn.Flatten()

    def forward(self, image):

        x = self.efficient.extract_features(image)
        x = self.conv1(x)
        x = self.conv2(x)
        fea = self.avg_pool(x)
        x = self.flatten(fea)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return fea, x  # F.softmax(x, dim=1)
