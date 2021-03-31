import json
import torch.utils.data as data
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import random

class Dataset(data.Dataset):

    def __init__(self, opt, train=True) -> None:
        super().__init__()
        self.train = train
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize([224, 224], Image.BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=1.)
        self.data_path = opt.data_root

        if train:
            with open(os.path.join(self.data_path, 'training_annotation.json'), 'r') as load_f:
                self.ann = json.load(load_f)
            with open(os.path.join(self.data_path, 'relationship.json'), 'r') as load_f:
                self.re = json.load(load_f)
                self.re_reverse = {}
                for key, value in self.re.items():
                    self.re_reverse[value] = key

            self.data_len = 447
        else:
            self.data_len = 119

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        images = []
        if self.train:
            name = 'train'
        else:
            name = 'test'
        path = os.path.join(self.data_path, name, name, str(index).zfill(6))
        if self.train:
            flip = random.random() > 0.5
            label = self.ann[str(index).zfill(6)]
        else:
            flip = False
        flat = None
        for file in sorted(os.listdir(path)):
            image = Image.open(os.path.join(path, file)).convert('RGB')
            if flip:
                if flat is None:
                    label, flat = self.get_flip_label(self.ann[str(index).zfill(6)])
                if flat:
                    image = self.flip(image)
            image = self.resize(image)
            images.append(torch.unsqueeze(self.totensor(image), dim=0))
        images = torch.cat(images, dim=0) * 2. - 1.
        if self.train:
            label = torch.LongTensor(label)
            object = F.one_hot(label[0], num_classes=35).int()
            relation = F.one_hot(label[1], num_classes=82).int()
            subject = F.one_hot(label[2], num_classes=35).int()
            return {'data': images, 'object': object, 'relation': relation, 'subject': subject}
        else:
            return {'data': images}

    def get_flip_label(self, label):

        if 'right' in self.re_reverse[label[1]]:
            label[1] = self.re[self.re_reverse[label[1]].replace('right', 'left')]
        elif 'left' in self.re_reverse[label[1]]:
            label[1] = self.re[self.re_reverse[label[1]].replace('left', 'right')]
        else:
            return label, False

        return label, True

