import torch
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar
import errno
import os
import sys
import time
import math
import torch.nn as nni
import torch.nn.init as init
from torch.autograd import Variable
import shutil
import numpy as np

def linear_rampup(current, rampup_length): 
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, lambda_u, num_epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = torch.mean(-torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, num_epoch)

class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * (1 - self.alpha))
                param.mul_(1- self.wd)   # weight decay

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

def save(result_dir, epoch, model, ema_model, val_acc, best_acc, is_best, optimizer):
    state = {
            'epoch' : epoch + 1,
            'state_dict' : model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'acc' : val_acc,
            'best_acc' : best_acc,
            'optimizer' : optimizer.state_dict()
            }
    filename = f'checkpoint{epoch}.pth.tar'
    filepath = os.path.join(result_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(result_dir, 'model_best.pth.tar'))

def load(checkpoint, model, ema_model, optimizer):
    assert os.path.isfile(checkpoint), 'Error : no checkpoint directory found'
    result_dir = os.path.dirname(checkpoint)
    checkpoint = torch.load(checkpoint)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    if model is not None:
        model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    return model, optimizer, ema_model, best_acc, start_epoch

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


