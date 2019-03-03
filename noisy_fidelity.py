from default_params import *
import torch as th
import numpy as np
from optical_nn import *

def get_fidelity(U0, U, normalize=False):
    D, _ = U.shape
    num = np.trace(U.H @ U0)
    if normalize:
        den = (D * np.trace(U.H @ U))**0.5
    else:
        den = D
    return np.abs(num / den)**2
def noisy_f_test_diag(net, f_out, sigma_list=NOISY_TEST_SIGMAS, n_trials=NOISY_TEST_TRIALS):
    U0 = net.get_U()
    fidelity = np.zeros((n_trials, len(sigma_list)))
    for i, sig in enumerate(sigma_list):
        net.sigma_BS = sig
        net.sigma_PS = sig
        print(f'sigma : {sig}')
        for n in range(n_trials):
            U = net.get_U()
            F = get_fidelity(U0, U)
            print(f'Trial {n}, Fidelity {F:.4f}')
            fidelity[n, i] = F
        print(fidelity[:, i].mean())
        print(np.std(fidelity[:, i]))

    np.save(f_out, fidelity)
    return fidelity

if __name__ == '__main__':

    D = 256
    grid_unitary = th.load(os.path.join(DIR_TRAINED_MODELS, 'GridUnitary_256.pth'))
    grid_unitary.approx_sigma_bs=False
    stacked_net = StackedFFTUnitary(256, n_stack=32)
    fft_net = FFTUnitary(256)
    trunc_net = Unitary.truncated(8)(256)

    f_unit = os.path.join(DIR_RESULTS, 'unitary_fidelity.npy')
    f_stack = os.path.join(DIR_RESULTS, 'stacked_fft_32_fid.npy')
    f_fft = os.path.join(DIR_RESULTS, 'fft_fid.npy')
    f_trunc = os.path.join(DIR_RESULTS, 'trunc_fid.npy')
    #noisy_f_test_diag(stacked_net, f_stack, n_trials=20)
    noisy_f_test_diag(trunc_net, f_trunc, n_trials=20)
    noisy_f_test_diag(fft_net, f_fft, n_trials=20)
