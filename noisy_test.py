import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from time import time
from torchvision import datasets, transforms
import complex_torch_var as ct
from time import time
from scipy.ndimage import uniform_filter
from mnist import *
from train_mnist import *
from default_params import *
import os


def noisy_test_diag(net, f_out, sigma_list=NOISY_TEST_SIGMAS, n_trials=NOISY_TEST_TRIALS):
    accuracies = np.zeros((n_trials, len(sigma_list)))
    for i, sig in enumerate(sigma_list):
        net.set_noise(sig, sig)
        print(f'sigma : {sig}')
        for n in range(n_trials):
            acc = get_acc(net)
            print(f'Trial {n}, Accuracy {acc:.4f}')
            accuracies[n, i] = acc
        print(accuracies[:, i].mean())
        print(np.std(accuracies[:, i]))

    np.save(f_out, accuracies)
    return accuracies
def noisy_test_bs_ps(net, f_out, sigma_list=NOISY_TEST_SIGMAS, n_trials=NOISY_TEST_TRIALS):
    accuracies = np.zeros((n_trials, len(sigma_list), len(sigma_list)))
    for i, sig_p in enumerate(sigma_list):
        for j, sig_b in enumerate(sigma_list):
            net.set_noise(sig_p, sig_b)
            print(f'sigma : {sig_p, sig_b}')
            for n in range(n_trials):
                acc = get_acc(net)
                accuracies[n, i, j] = acc
                print(f'Trial {n}, Accuracy {acc:.4f}')
            mean = accuracies[:, i, j].mean()
            std = np.std(accuracies[:, i, j])
            print(f'Accuracy mean {mean:.4f}, stdev {std:.4f}')

    np.save(f_out, accuracies)
    return accuracies

def extract_diag_fft():
    acc_psbs = np.load(F_FFT_ACC_PSBS)
    acc_diag = np.array([np.diag(x) for x in acc_psbs])
    np.save(F_FFT_ACC_DIAG, acc_diag)
def extract_diag_grid():
    acc_psbs = np.load(F_GRID_ACC_PSBS)
    acc_diag = np.array([np.diag(x) for x in acc_psbs])
    np.save(F_GRID_ACC_DIAG, acc_diag)

def test_grid():
    grid_net = load_grid()
    noisy_test_diag(grid_net, f_out=F_GRID_ACC_DIAG)
def test_grid_ordered_sv():
    grid_net = load_grid(rand_S=False)
    noisy_test_diag(grid_net, f_out=F_GRID_ORD_ACC_DIAG)
def test_fft(diag=True):
    fft_net = load_fft()
    if diag:
        noisy_test_diag(fft_net,  f_out=F_FFT_ACC_DIAG)
    else:
        noisy_test_bs_ps(fft_net,  f_out=F_FFT_ACC_PSBS)
if __name__ == '__main__':

    assert False
    #fft_net = load_fft().to(DEVICE)
    #grid_net = load_grid().to(DEVICE)
    net = th.load(os.path.join(DIR_TRAINED_MODELS, 'stacked_fft_1.pth'))
    print(get_acc(net))
    noisy_test_diag(net, f_out=os.path.join(DIR_NOISY_TEST, 'stacked_fft_1.npy'))

    assert False
    net = th.load(os.path.join(DIR_TRAINED_MODELS, 'stacked_fft_32.pth'))
    print(get_acc(net))
    noisy_test_diag(net, f_out=os.path.join(DIR_NOISY_TEST, 'stacked_fft_diag.npy'))

    assert False
    cgrd_net = mnist_ONN(unitary=CGRDUnitary)
    cgrd_net.load_state_dict( th.load(os.path.join(DIR_TRAINED_MODELS, 'cgrd.pth')))
    noisy_test_diag(cgrd_net, f_out=os.path.join(DIR_NOISY_TEST, 'CGRD_diag.npy'))



    #noisy_test_bs_ps(fft_net, f_out=F_FFT_ACC_PSBS)
    #noisy_test_bs_ps(grid_net, f_out=F_GRID_ACC_PSBS)
    #test_grid_ordered_sv()
