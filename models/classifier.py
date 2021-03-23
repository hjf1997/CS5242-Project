import time
import csv
import torch
import torchvision
from torch.nn import init
from models.network import Vgg
import torch.nn as nn
from models.base_model import BaseModel
from coco_name import coco_name
import os
import json
import random

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            if m.affine:
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # init_weights(net, init_type, init_gain=init_gain)
    return net

class Model(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.gpu_ids = opt.gpu_ids
        self.model_names = ['Cla']
        self.loss_names = ['bce']

        self.netCla = Vgg()
        self.netCla = init_net(self.netCla, gpu_ids=self.gpu_ids)
        if not opt.test:
            self.optm_net = torch.optim.Adam(self.netCla.parameters(), lr=2e-4)
            self.loss = nn.BCELoss().to(self.device)

        with open(os.path.join(opt.data_root, 'object1_object2.json'), 'r') as load_f:
            self.obj_ = json.load(load_f)
            self.obj = {}
            for key, value in self.obj_.items():
                self.obj[value] = key

    def set_input(self, input):

        self.images = input['data'].to(self.device)
        if self.isTrain:
            self.object = input['object'].to(self.device)
            self.subject = input['subject'].to(self.device)

    def train(self, train_loader, opt):

        self.isTrain = True
        self.setup(opt)  # regular setup: load and print networks; create schedulers
        total_iters = 0
        for epoch in range(opt.n_epochs+1):
            epoch += 1
            iter_data_time = time.time()  # timer for data loading per iteration
            for i, data in enumerate(train_loader):
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_iters += 1

                self.set_input(data)
                predictions = self.netCla(self.images[:, random.randint(0, 29)])

                target = (self.object | self.subject).float()
                self.loss_bce = self.loss(predictions, target)
                loss_ = self.loss_bce
                self.optm_net.zero_grad()
                loss_.backward()
                self.optm_net.step()

                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = self.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    self.print_current_losses(epoch, total_iters, losses, t_comp, t_data)

                if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    self.save_networks(save_suffix)

                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                self.save_networks('latest')
                self.save_networks(epoch)



    def test(self, test_loader, opt):
        self.isTrain = False
        self.setup(opt)  # regular setup: load and print networks; create schedulers

        out = open('prediction.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        label = ['Id', 'label']
        csv_write.writerow(label)
        count = 0
        for i, data in enumerate(test_loader):
            self.set_input(data)
            with torch.no_grad():
                # process object
                object, relation, subject = self.netC(self.images)
                pred_object = object.topk(5, sorted=True)[1].cpu().tolist()[0]
                csv_write.writerow([count, " ".join([str(x) for x in pred_object])])
                count += 1
                pred_relation = relation.topk(5, sorted=True)[1].cpu().tolist()[0]
                csv_write.writerow([count, " ".join([str(x) for x in pred_relation])])
                count += 1
                pred_subject = subject.topk(5, sorted=True)[1].cpu().tolist()[0]
                csv_write.writerow([count, " ".join([str(x) for x in pred_subject])])
                count += 1
        out.close()


    def cal_top5(self, pred, target):
        score = 0
        _, target = torch.where(target == 1)
        target = target.unsqueeze(-1)
        _, pred_ = pred.topk(5, sorted=True)
        _, index = torch.where(target == pred_)
        index = index.cpu().tolist()
        for i in index:
            if i == 0:
                score += 1.
            elif i == 1:
                score += 0.5
            elif i == 2:
                score += 0.33
            elif i == 3:
                score += 0.25
            elif i == 4:
                score += 0.2
        return score / pred.shape[0]
