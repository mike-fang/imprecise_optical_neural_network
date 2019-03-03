import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import uniform_filter
from scipy import stats
from train_mnist import *
from default_params import *

def compare_plot(accs, names, log_plot=False, f_name=None, q='Accuracy', acc_loss=False, colors=None):
    if acc_loss:
        accs = [a - a[0, 0] for a in accs]
        plt.ylabel('Change in Accuracy', fontsize=16)
    else:
        plt.ylabel('Classification Accuracy', fontsize=16)
    if log_plot:
        plot = plt.semilogy
    else:
        plot = plt.plot

    sigs = []
    for acc in accs:
        sigs.append(
                np.linspace(0, 0.02, acc.shape[1])
                )

    if colors is None:
        colors = ['r', 'b', 'g', 'purple', 'orange']
    for n in range(len(accs)):
        acc = accs[n]
        name = names[n]
        sig = sigs[n]
        c = colors[n]
        med = np.quantile(acc, 0.5, axis=0)
        high = np.quantile(acc, 0.8, axis=0)
        low = np.quantile(acc, 0.2, axis=0)
        plot(sig, med, color=c, label=f'Median {q} ({name})')
        plot(sig, high, color=c, linestyle='--', label=f'20/80% Quantile ({name})')
        plot(sig, low, color=c, linestyle='--')
        plt.fill_between(sig, low, high, color=c, alpha=.2)

    plt.xlim([0, 0.02])
    plt.xticks([0, 0.005, 0.01, 0.015, 0.02], fontsize=14)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 0.02, 0.0025), minor=True)
    plt.yticks( fontsize=14)
    plt.xlabel(r'Component Error, $\sigma_{PS} = \sigma_{BS}$', fontsize=16)
    plt.grid(linestyle=':', which='both')
    plt.legend(fontsize=14)


    plt.tight_layout()
    if f_name:
        plt.savefig(f_name)
def plot_psbs_acc(accuracies):
    mean = accuracies[:, :, ::-1].mean(0).T
    mean = accuracies.mean(0).T
    extent=[0, NOISY_TEST_MAX, 0.00, NOISY_TEST_MAX]
    plt.imshow(mean, extent=extent, cmap='inferno', origin='lower')
    cbar = plt.colorbar()
    ctr = plt.contour(uniform_filter(mean, 3), colors='white', extent=extent, linewidths=4)
    cbar.add_lines(ctr)
    
    plt.xticks([0, 0.005, 0.01, 0.015, 0.02], fontsize=14)
    plt.yticks([0, 0.005, 0.01, 0.015, 0.02], fontsize=14)
    plt.ylabel(r'Beamsplitter Error ($\sigma_{BS}$)', fontsize=16)
    plt.xlabel(r'Phaseshifter Error ($\sigma_{PS}$)', fontsize=16)
    #plt.title('Classification Accuracy of Noisy Grid Network', fontsize=18, y=1.08)
    plt.tight_layout()

def plot_fft_psbs(f=F_FIG_FFT_PSBS):
    plt.cla()
    plt.clf()
    accuracies = np.load(F_FFT_ACC_PSBS)
    plot_psbs_acc(accuracies)
    plt.plot([0, 0.02], [0, 0.02], 'r--', linewidth=2)
    plt.savefig(f)
def plot_grid_psbs(f=F_FIG_GRID_PSBS):
    plt.cla()
    plt.clf()
    accuracies = np.load(F_GRID_ACC_PSBS)
    plot_psbs_acc(accuracies)
    plt.plot([0, 0.02], [0, 0.02], 'b--', linewidth=2)
    plt.savefig(f)
def plot_grid_vs_fft():
    plt.cla()
    plt.clf()
    acc_grid = np.load(F_GRID_ACC_DIAG)
    acc_fft = np.load(F_FFT_ACC_DIAG)
    compare_plot([acc_grid, acc_fft], ['GridNet', 'FFTNet'])
    plt.savefig(F_FIG_COMPARE_GRID_FFT)
def plot_rand_vs_ordered():
    plt.cla()
    plt.clf()
    acc_rand = np.load(F_GRID_ACC_DIAG)
    acc_order = np.load(F_GRID_ORD_ACC_DIAG)
    compare_plot([acc_rand, acc_order], ['Random SV', 'Ordered SV'])
    plt.savefig(F_FIG_RAND_VS_ORD)
def plot_compare_fft_diff_nh():
    plt.cla()
    plt.clf()
    accs = []
    names = []
    dir = os.path.join(DIR_NOISY_TEST, 'fft_net_diff_nh')
    files = os.listdir(dir)
    files = sorted(files, key=lambda f_name:int(f_name[:-4]), reverse=True)
    print(files)
    for f in files:
        f_path = os.path.join(dir, f)
        accs.append(
                np.load(f_path)
                )
        names.append(
                f'FFTNet, D={f[:-4]}'
                )
    compare_plot(accs, names)
def plot_fft_diff_depth():
    plt.cla()
    plt.clf()
    accs = []
    names = []
    dir = os.path.join(DIR_RESULTS, 'fft_net_diff_depth')
    files = os.listdir(dir)
    files = sorted(files, key=lambda f_name:int(f_name[:-4]), reverse=True)
    print(files)
    for f in files:
        f_path = os.path.join(dir, f)
        accs.append(
                np.load(f_path)
                )
        names.append(
                f'FFTNet, D={f[:-4]}'
                )
    compare_plot(accs, names)
def truncated_vs_fft():
    f_out_trunc = os.path.join(DIR_RESULTS, 'trunc_fid.npy')
    f_out_fft = os.path.join(DIR_RESULTS, 'fft_fid.npy')

    fid_trunc = np.load(f_out_trunc)
    fid_fft = np.load(f_out_fft)
    
    compare_plot([fid_trunc, fid_fft], ['TruncGrid', 'FFTUnitary'], acc_loss=False, q='Fidelity', colors=[(1, .3, 0), 'blue'])
    plt.ylabel('Fidelity')
    plt.savefig(os.path.join(DIR_FIGS, 'truncated_vs_fft.pdf'))
    plt.show()
def stacked_vs_grid():
    f_out_grid = os.path.join(DIR_RESULTS, 'unitary_fidelity.npy')
    f_out_fft_32 = os.path.join(DIR_RESULTS, 'stacked_fft_32_fid.npy')

    fid_grid = np.load(f_out_grid)
    fid_fft_32 = np.load(f_out_fft_32)
    
    compare_plot([fid_grid, fid_fft_32], ['GridUnitary', 'StackedFFT'], acc_loss=False, q='Fidelity', colors=['red', 'green'])
    plt.ylabel('Fidelity')
    plt.savefig(os.path.join(DIR_FIGS, 'stacked_vs_grid_fidelity.pdf'))
    plt.show()

if __name__ == '__main__':

    plot_rand_vs_ordered()

    assert False
    acc_grid = np.load(F_GRID_ACC_DIAG)
    acc_stacked_fft = np.load(
            os.path.join(DIR_NOISY_TEST, 'stacked_fft_diag.npy')
            )
    acc_layer_1 = np.load(
            os.path.join(DIR_NOISY_TEST, 'grid_1_layer.npy')
            )
    acc_stacked_1 = np.load(
            os.path.join(DIR_NOISY_TEST, 'stacked_fft_1.npy')
            )
    compare_plot([acc_grid, acc_stacked_fft], ['GridNet', 'StackedFFT-Net'], acc_loss=False, colors=['red', 'green'])
    plt.savefig(os.path.join(DIR_FIGS, 'stacked_vs_grid_accuracy.pdf'))
    plt.show()

    assert False

    acc_grid = np.load(F_GRID_ACC_DIAG)
    #acc_grid_ord = np.load(F_GRID_ORD_ACC_DIAG)
    #acc_fft_ = np.load(F_FFT_ACC_DIAG + '_')
    acc_fft = np.load(F_FFT_ACC_PSBS)
    acc_fft = acc_fft[:, np.arange(21), np.arange(21)]
    #acc_fft = np.array([np.diag(acc) for acc in acc_grid])

    #acc_fft = np.load(F_FFT_ACC_DIAG)
    sigs = NOISY_TEST_SIGMAS
    compare_plot(acc_grid, acc_fft, 'FFT_', 'FFT')
    plt.show()

    assert False
    f_name = './figures/acc_compare.pdf'
    acc_grid = np.load('./noisy_grid_infr_256/accuracies.npy')
    acc_grid = np.array([
        np.diag(acc) for acc in acc_grid
        ])
    acc_fft = np.load('./noisy_fft_93/accuracies.npy')
    acc_fft = np.array([
        np.diag(acc) for acc in acc_fft
        ])
    sigs_1 = np.linspace(0, 0.02, 20).tolist()
    sigs_2 = np.linspace(0, 0.02, 21).tolist()

    acc_rand = np.load('./noisy_grid_infr_256/shuffled_accuracies.npy')
    acc_hyb = np.load('hybrid_accuracies.npy')
    acc_grid_2 = np.load('./results/noisy_test/grid_net_diag.npy')

    #compare_plot(acc_1, acc_2, False, f_name='./figures/acc_compare.pdf')
    #compare_plot(acc_rand, acc_grid, 'Shuffled', 'Ordered', False, f_name='./figures/compare_permute.pdf', q='Acc. Loss', acc_loss=True)
    #compare_plot(acc_rand, acc_hyb, 'Grid', 'BlockFFT', False, f_name='./figures/compare_hybrid.pdf', q='Acc. Loss', acc_loss=True)
    compare_plot(acc_rand, acc_grid_2, 'Grid', 'Grid_2', False, f_name='./figures/compare_grid.pdf', q='Accuracy')
    #compare_plot(acc_1, acc_2, 'Grid', 'BlockFFT', False, f_name='./figures/compare_hybrid.png')
    plt.ylabel('Accuracy Loss', fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig('./figures/compare_hybrid.pdf')
    #plt.ylim([0.6, 1])
    plt.show()
    assert False
    err_1, err_fit, m, b = get_fit_err(sigs_1, acc_1)
    err_2, err_fit2, m2, b2 = get_fit_err(sigs_2, acc_2)
    compare_plot(err_1, err_2, sigs_1, sigs_2, log_plot=True, f_name=None)
    label = r'$\epsilon \approx %.3f \times e^{ %d \sigma }$' % (np.exp(b), m)
    plt.plot(sigs_1, err_fit, 'm.', label=label)
    label = r'$\epsilon \approx %.3f \times e^{ %d \sigma }$' % (np.exp(b2), m2)
    plt.plot(sigs_2, err_fit2, 'c.', label=label)

    plt.show()
