import numpy as np
import torch as th
import matplotlib.pylab as plt
from optical_nn import *
import complex_torch_var as ct
from mnist import *
import os
from time import time
from functools import partial
from glob import glob
from default_params import *

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Good learning rates for different networks
LR_FFT = 5e-2
LR_GRID = 2.5e-4
LR_COMPLEX = 5e-3

def train_complex(f=F_COMPLEX_TRAIN, n_h=[256, 256]):
    train_params = {}
    train_params['n_epochs'] = 5
    train_params['log_interval'] = 100
    train_params['batch_size'] = 100

    optim_params = {}
    optim_params['lr'] = 5e-3
    optim_params['momentum'] = .9

    net = mnist_complex(hidden_units=n_h)
    print(net)
    train(net, **train_params, optim_params=optim_params)
    optim_params['lr'] /= 5
    train(net, **train_params, optim_params=optim_params)
    acc = get_acc(net)
    print(f'Trained ComplexNet with accuracy {acc}.')
    if f:
        th.save(net.state_dict(), f)
        print(f'Saved model to {f}.')
def train_cgrd(f=F_CGRD_TRAIN):
    train_params = {}
    train_params['n_epochs'] = 5
    train_params['log_interval'] = 100
    train_params['batch_size'] = 100

    optim_params = {}
    optim_params['lr'] = LR_GRID
    optim_params['momentum'] = .9

    net = mnist_ONN(unitary=CGRDUnitary)
    if f:
        th.save(net.state_dict(), f)
        print(f'Saved model to {f}.')
    if f:
        th.save(net.state_dict(), f)
        print(f'Saved model to {f}.')
    train(net, **train_params, optim_params=optim_params)
    optim_params['lr'] /= 5
    train(net, **train_params, optim_params=optim_params)
    acc = get_acc(net)
    print(f'Trained ComplexNet with accuracy {acc}.')
    if f:
        th.save(net.state_dict(), f)
        print(f'Saved model to {f}.')
def train_fft(f=F_FFT_TRAIN, n_h=[256, 256]):
    train_params = {}
    train_params['n_epochs'] = 5
    train_params['log_interval'] = 100
    train_params['batch_size'] = 100

    optim_params = {}
    optim_params['lr'] = LR_FFT * 3
    optim_params['momentum'] = .9

    net = mnist_ONN(FFTUnitary, hidden_units=n_h)
    print(net)
    train(net, **train_params, optim_params=optim_params)
    optim_params['lr'] /= 5
    train(net, **train_params, optim_params=optim_params)
    acc = get_acc(net)
    print(f'Trained FFTNet with accuracy {acc}.')
    if f:
        th.save(net, f)
        print(f'Saved model to {f}.')
def convert_save_grid_net(complex_net=None, f=None, rand_S=True):
    if complex_net is None:
        complex_net = load_complex()
    if f is None:
        f = F_GRID_TRAIN if rand_S else F_GRID_ORD_TRAIN
    grid_net = complex_net.to_grid_net(rand_S=rand_S).to(DEVICE)
    acc = get_acc(grid_net)
    print(f'Converted to GridNet with accuracy {acc} with {"shuffled" if rand_S else "ordered"} singular values.')
    th.save(grid_net.state_dict(), f)
    print(f'Saved GridNet at {f}')
def batch_train_complex(n_train, dir = DIR_COMPLEX_TRAIN):
    for _ in range(n_train):
        f = os.path.join(dir, f'{time():.0f}')
        train_complex(f=f)
def batch_convert(dir = DIR_COMPLEX_TRAIN):
    for f in glob(os.path.join(dir, '*')):
        net = load_complex(f)
        convert_save_grid_net(net, f=f+'_grid')
def load_complex(f=F_COMPLEX_TRAIN):
    net = mnist_complex()
    net.load_state_dict(th.load(f, map_location=DEVICE))
    acc = get_acc(net)
    print(f'ComplexNet loaded from {f} with accuracy {acc}.')
    return net.to(DEVICE)
def load_grid(f=None, rand_S=True, report_acc=True):
    if f is None:
        f = F_GRID_TRAIN if rand_S else F_GRID_ORD_TRAIN
    net = mnist_ONN()
    net.load_state_dict(th.load(f, map_location=DEVICE))
    if report_acc:
        acc = get_acc(net)
        print(f'GridNet loaded from {f} with accuracy {acc}.')
    else:
        print(f'GridNet loaded from {f}.')
    return net.to(DEVICE)
def load_fft(f=F_FFT_TRAIN):
    net = mnist_ONN(FFTUnitary)
    net.load_state_dict(th.load(f, map_location=DEVICE))
    acc = get_acc(net)
    print(f'FFTNet loaded from {f} with accuracy {acc}.')
    return net.to(DEVICE)

if __name__ == '__main__':

    net = load_fft()
    
    for data, target in mnist_loader(train=False, batch_size=100, shuffle=False):
        continue
    data = data.view(-1, 28**2)
    data, target = data.to(DEVICE), target.to(DEVICE)
    print(th.max(net(data), dim=1))
