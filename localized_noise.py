import matplotlib
matplotlib.use("TkAgg")
from mnist import *
from optical_nn import *
from train_mnist import *
import complex_torch_var as ct
import numpy as np
import torch as th
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mnist import *
from torch import optim
from glob import iglob
from scipy import optimize
import os
import matplotlib.gridspec as gridspec
from default_params import *
LAYER = 2

def load_model(use_fft):
    if not use_fft:
        net = optical_net()
        f_path = './nl_optical_256.pth'
    else:
        net = fft_net()
        f_path = './fft_mnist/nl_train.pth'

    net.load_state_dict(th.load(f_path, map_location=DEVICE))
    return net
def grid_phase_mat(net, checkered=False):
    for n, p in net.named_parameters():
        if n == 'theta_A':
            phase = p.data.cpu().numpy()
            nLA, nDA = phase.shape
            theta_a  = ((phase + np.pi) % (np.pi * 2)) - np.pi
        if n == 'theta_B':
            phase = p.data.cpu().numpy()
            nLB, nDB = phase.shape
            theta_b = np.zeros((nLB, nDA), dtype=theta_a.dtype)
            theta_b[:, :nDB]  = ((phase + np.pi) % (np.pi * 2)) - np.pi

    theta = np.zeros((nLA*2, nDA*2), dtype=theta_a.dtype)
    theta[0::2, 0::2] = theta_a
    theta[1::2, 1::2] = theta_b
    if not checkered:
        theta[0::2, 1::2] = theta_a
        theta[1::2, 2::2] = theta_b[:, :-1]
    return theta
def central_band_std(refls, delta=10):
    horz_cut_refls = (refls[:, 128-delta:128+delta:2, :]).copy()
    horz_cut_refls += (refls[:, 128-delta+1:128+delta:2, :])
    horz_cut_refls = horz_cut_refls.reshape((-1, horz_cut_refls.shape[-1]))

    horz_cut_refl_std = np.std(horz_cut_refls, axis=0)
    return horz_cut_refl_std[:-1]
    plt.title(r'Distribution of Internal Phaseshift ($\theta_{m, l}$)')
    plt.plot(np.arange(256-1), horz_cut_relf_std[:-1])
def get_theta_refl(f_list):
    refls = []
    thetas = []
    for f in f_list:
        net.load_state_dict(th.load(f, map_location=DEVICE))
        theta = grid_phase_mat(net[2].U, checkered=False)
        refl = np.sin(theta/2)

        refls.append(refl)
        thetas.append(theta)

    return np.array(thetas), np.array(refls)
def plot_MZIs(refl, name=False, full=True):
    fig, ax = plt.subplots()
    im = ax.imshow(refl, vmin=.0, vmax=np.pi/2, cmap='inferno')
    if full:
        plt.title(r'Distribution of MZI phase in $U_2$', fontsize=14)
        plt.xlabel('Layer depth (l)', fontsize=14)
        plt.ylabel('Dimension (d)', fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=[0, 0.25 * np.pi, 0.5 * np.pi])
        cbar.ax.set_yticklabels(
                ['0', r'$\pi/4$', r'$\geq\pi/2$']
                )
    else:
        ax.set_yticks(np.arange(refl.shape[0]), minor=True)
        ax.set_yticks(np.arange(refl.shape[0], 5), minor=False)
        ax.set_xticks(np.arange(refl.shape[0]), minor=True)
        ax.set_xticks(np.arange(refl.shape[0], 5), minor=False)

        ax.set_xticklabels(ax.xaxis.get_majorticklabels())

        ax.grid(which='minor', linestyle='-', linewidth=0.5, color='white', alpha=0.4)
    if full:
        fig.savefig(f'./grid_phase_pos.pdf', dpi=400)
    else:
        fig.savefig(f'./grid_phase_pos_{name}.pdf', dpi=400)
def make_hist():
    N_BINS = 50
    for n, idx in regions.items():
        if n == 'center':
            c = 'red'
        elif n == 'edge':
            c = 'lime'
        else:
            c = 'blue'
        thetas_crop = thetas[:, idx[0], idx[1]]
        thetas_flat = thetas_crop.flatten()
        plt.hist(thetas_flat, bins=N_BINS, range=(0, np.pi/2), density=True, color=c, alpha=.7)
        plt.xticks(
                [x * np.pi for x in [0, 1/8, 1/4, 3/8, 1/2]],
                [ r'0', r'', r'$\pi/4$', r'', r'$\pi/2$', ]
                )
    plt.xlabel(r'Internal Phase Shift ($\theta$)', fontsize=14)
    plt.ylabel(r'Normalized Frequency', fontsize=14) 
    plt.title(r'Distribution of Phase Shift in $U_2$ of GridNet', fontsize=14)
    plt.savefig(f'./grid_phase_hist.pdf')
def plot_local_sensitivity(U_mat, V_mat, trans, f_out=None, color_scale=1):
    s = color_scale
    vmax = max(U_mat.max(), V_mat.max())
    vmin = min(U_mat.min(), V_mat.min())
    if vmax > -vmin:
        vmax, vmin = s*vmax, -s*vmax
    else:
        vmax, vmin = -s*vmin, s*vmin

    #Make subplot grid
    plt.figure(figsize=(17, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[3, .8, 3, .3])
    VH_ax = plt.subplot(gs[0, 0])
    U_ax = plt.subplot(gs[0, 2])
    S_ax = plt.subplot(gs[0, 1])
    cbar_ax = plt.subplot(gs[0, 3])

    cmap = 'PiYG'
    U_im = U_ax.imshow(U_mat.T, cmap=cmap, vmin=vmin, vmax=vmax)
    VH_im = VH_ax.imshow(V_mat.T, cmap=cmap, vmin=vmin, vmax=vmax)
    S_ax.plot(trans, np.arange(256), 'k')
    S_ax.set_ylim([0, 255])
    S_ax.invert_yaxis()
    
    central_ticks = [0,  8,  40,  72,  104, 136, 168, 200, 232]

    VH_ax.set_xticklabels(central_ticks)
    VH_ax.set_yticklabels(central_ticks)
    U_ax.set_xticklabels(central_ticks)
    U_ax.set_yticklabels(central_ticks)
    S_ax.set_yticks(central_ticks)

    VH_ax.set_title(r'$V_2^\dagger$', fontsize=20)
    VH_ax.set_xlabel('Layer Depth (l)', fontsize=14)
    VH_ax.set_ylabel('Dimension (n)', fontsize=14)

    U_ax.set_title(r'$U_2$', fontsize=20)
    U_ax.set_xlabel('Layer Depth (l)', fontsize=14)

    S_ax.set_title(r'$\Sigma_2$', fontsize=20)
    S_ax.set_xlabel('Transmissivity', fontsize=14)

    plt.colorbar(VH_im, cax=cbar_ax)
    cbar_ax.yaxis.set_label_position('left')
    cbar_ax.set_ylabel('Accuracy Change', fontsize=14)

    if f_out:
        plt.savefig(f_out)
def calculate_local_sensitivity(net, U, f_name=None, sig=1e-1, block_size=8):
    perf_acc = get_acc(net)
    #Generate noise 
    theta_A = U.theta_A.clone()
    theta_B = U.theta_B.clone()
    mask_A = th.zeros_like(theta_A)
    mask_B = th.zeros_like(theta_B)
    noise_A = th.zeros_like(theta_A)
    noise_B = th.zeros_like(theta_B)
    noise_A.normal_()
    noise_B.normal_()

    N = U.D//2
    acc_mat = np.zeros((N//block_size, N//block_size))
    for i in range(N//block_size):
        for j in range(N//block_size):
            print(i, j)
            idx = (
                    slice((i)*block_size,(i+1)*block_size), slice((j)*block_size,(j+1)*block_size)
                    )
            mask_A *= 0
            mask_B *= 0
            mask_A[idx] = 1
            mask_B[idx] = 1
            U.theta_A = Parameter(theta_A + sig * noise_A * mask_A)
            U.theta_B = Parameter(theta_B + sig * noise_B * mask_B)

            #theta = grid_phase_mat(net[2].VH)
            #plt.imshow(theta)
            #plt.show()
            acc = get_acc(net) - perf_acc
            acc_mat[i, j] = acc
            print(acc)

    if f_name:
        np.save(f_name, acc_mat)
    return acc_mat
def make_plot(rand_S=True, s=1, layer_n=LAYER):
    net = load_grid(rand_S=rand_S)
    theta = net[layer_n].S.theta
    trans = (th.sin(theta/2)**2).cpu().data.numpy()
    if rand_S:
        U_name = F_LN_U_RAND
        V_name = F_LN_V_RAND
        f_out = F_FIG_LN_RAND
    else:
        U_name = F_LN_U_ORD
        V_name = F_LN_V_ORD
        f_out = F_FIG_LN_ORD

    U_mat = np.load(U_name)
    V_mat = np.load(V_name)
    print(f_out)
    plot_local_sensitivity(U_mat, V_mat, trans, f_out, color_scale=s)
def get_mats(rand_S=True, layer_n=LAYER):
    net = load_grid(rand_S=rand_S)
    layer = net[layer_n]
    VH = layer.VH
    U = layer.U
    if rand_S:
        U_name = F_LN_U_RAND
        V_name = F_LN_V_RAND
    else:
        U_name = F_LN_U_ORD
        V_name = F_LN_V_ORD
    calculate_local_sensitivity(net, U, U_name, sig=1e-1)
    calculate_local_sensitivity(net, VH, V_name, sig=1e-1)

def calculate_local_sensitivity_fft(net, U, f_name=None, sig=1e-1, block_size=8):
    perf_acc = get_acc(net)

    #Generate noise 
    theta = U.theta.clone()
    mask = th.zeros_like(theta)
    noise = th.zeros_like(theta)
    noise.normal_()

    N, M = theta.shape
    acc_mat = np.zeros((N//block_size, M//block_size))
    for i in range(N//block_size):
        for j in range(M//block_size):
            print(i, j)
            idx = (
                    slice((i)*block_size,(i+1)*block_size), slice((j)*block_size,(j+1)*block_size)
                    )
            mask *= 0
            mask[idx] = 1
            U.theta = Parameter(theta + sig * noise * mask)
            acc = get_acc(net) - perf_acc
            acc_mat[i, j] = acc
            print(acc)
    if f_name:
        np.save(f_name, acc_mat)
    return acc_mat
def get_mats_fft(layer_n=LAYER):
    net = load_fft()
    layer = net[layer_n]
    U = layer.U
    VH = layer.VH
    U_name = F_LN_U_FFT
    V_name = F_LN_V_FFT
    calculate_local_sensitivity_fft(net, U, U_name, sig=1e-1)
    calculate_local_sensitivity_fft(net, VH, V_name, sig=1e-1)
def make_plot_fft(s=1, layer_n=LAYER):
    net = load_fft()
    theta = net[layer_n].S.theta
    trans = (th.sin(theta/2)**2).cpu().data.numpy()
    U_name = F_LN_U_FFT
    V_name = F_LN_V_FFT
    f_out = F_FIG_LN_FFT

    U_mat = np.load(U_name)
    V_mat = np.load(V_name)
    plot_local_sensitivity(U_mat, V_mat, trans, f_out, color_scale=s)
if __name__ == '__main__':
    net = load_fft()
    layer = net[LAYER]
    U = layer.U
    print(U)

    make_plot_fft()

    assert False
    get_mats_fft()

    get_mats(True)
    get_mats(False)
    make_plot(True)
    make_plot(False, 0.4)
