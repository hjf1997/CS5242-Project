import argparse
from dataset import Dataset
import torch
from torch.utils.data import DataLoader
from models.model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/junfeng/dataset/5242')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--gpu_ids', type=str, default="7")
    parser.add_argument('--name', type=str, default='model',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--load_iter', type=int, default='0',
                        help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--save_epoch_freq', type=int, default=5,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--test_epoch_freq', type=int, default=20,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--n_threads', type=int, default=6)
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--test', action='store_true')
    opt = parser.parse_args()

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])


    model = Model(opt)
    if not opt.test:
        train_dataloader = DataLoader(Dataset(opt), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_threads)
        model.train(train_dataloader, opt)
    else:
        test_dataloader = DataLoader(Dataset(opt, train=False))
        model.test(test_dataloader, opt)
