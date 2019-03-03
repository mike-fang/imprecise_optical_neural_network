import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from time import time
from torchvision import datasets, transforms
import complex_torch_var as ct
from time import time
from mnist import *
from default_params import *
from train_mnist import *

def noisy_prob_test(net, n_trials, sigma_BS, sigma_PS, f_name):
    noisy_prob = np.zeros((n_trials, 10))
    net.set_noise(sigma_PS, sigma_BS)

    for n in range(n_trials):
        prob = (th.exp(net(X0))).data.cpu().numpy()[0]
        noisy_prob[n] = prob

    if f_name:
        np.save(f_name, noisy_prob)

    return noisy_prob
def noisy_prob_plot(noisy_prob, title=None, f_name=None):
    print(f_name)
    print(noisy_prob)
    
    med_prob = np.quantile(noisy_prob, 0.5, axis=0)
    low_prob = np.quantile(noisy_prob, 0.2, axis=0)
    high_prob = np.quantile(noisy_prob, 0.8, axis=0)
    mean_prob = noisy_prob.mean(axis=0)
    no_err = np.all(low_prob == high_prob)

    err_h = high_prob - med_prob
    err_l = med_prob - low_prob

    plt.xticks(np.arange(10), fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0,1])
    print(med_prob)

    if title:
        plt.title(title, fontsize=18)

    if no_err:
        plt.bar(list(range(10)), mean_prob, color='white', edgecolor='black')
    else:
        plt.bar(list(range(10)), mean_prob, yerr=[err_l, err_h], capsize=5, color='white', edgecolor='black')
    plt.ylabel('Predication Probability', fontsize=16)
    plt.xlabel('Class', fontsize=16)

    if f_name:
        plt.savefig(f_name)

if __name__ == '__main__':

    USE_FFT = True
    if USE_FFT:
        net = load_fft()
        arch_name = 'FFT'
    else:
        net = load_grid()
        arch_name = 'grid'
    n_trials = 20
    sigma_PS = 0.01 * 1
    sigma_BS = 0.01 * 1 
    dir = DIR_FIGS
    for sigma_PS, sigma_BS, suff in [(0, 0, ''), (0.01, 0.01, '_PS_BS')]:
        prob = noisy_prob_test(net, n_trials, sigma_BS * 2, sigma_PS, f_name=None)
        if suff == '':
            title = f'Output with Ideal Components ({arch_name})'
        elif suff == '_PS_BS': 
            title = f'Output with Imprecise Components ({arch_name})'
        f_out = os.path.join(DIR_FIGS, f'{arch_name}{suff}.pdf')
        noisy_prob_plot(prob, title=title, f_name=f_out)
        plt.cla()
        plt.clf()
        plt.close()
