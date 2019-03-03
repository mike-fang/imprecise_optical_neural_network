import os
import numpy as np
# Default file paths
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


# Trained Models
DIR_TRAINED_MODELS = os.path.join(DIR_PATH, 'trained_models')
F_COMPLEX_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'complex_net.pth')
F_CGRD_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'cgrd.pth')
F_GRID_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'grid_net.pth')
F_GRID_ORD_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'grid_net_ordered_SV.pth')
F_FFT_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'fft_net.pth')

DIR_COMPLEX_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'complex_net')
DIR_GRID_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'grid_net')
DIR_GRID_ORD_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'grid_ord_net')
DIR_FFT_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'fft_net')
DIR_FFT_NH_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'fft_net_diff_nh')
DIR_FFT_DEPTH_TRAIN = os.path.join(DIR_TRAINED_MODELS, 'fft_net_diff_depth')

# Good learning rates for different networks
LR_FFT = 5e-2
LR_GRID = 2.5e-4
LR_COMPLEX = 5e-3

# Noisy Test Acc
DIR_NOISY_TEST = os.path.join(DIR_PATH, 'results', 'noisy_test')
F_GRID_ACC_DIAG = os.path.join(DIR_NOISY_TEST, 'GridNet_diag.npy')
F_GRID_ORD_ACC_DIAG = os.path.join(DIR_NOISY_TEST, 'GridNet_ordered_SV_diag.npy')
F_FFT_ACC_DIAG = os.path.join(DIR_NOISY_TEST, 'FFTNet_diag.npy')
F_GRID_ACC_PSBS = os.path.join(DIR_NOISY_TEST, 'GridNet_psbs.npy')
F_GRID_ORD_ACC_PSBS = os.path.join(DIR_NOISY_TEST, 'GridNet_ordered_SV_psbs.npy')
F_FFT_ACC_PSBS = os.path.join(DIR_NOISY_TEST, 'FFTNet_psbs.npy')


NOISY_TEST_MAX = 0.02
NOISY_TEST_SIGMAS = np.linspace(0, NOISY_TEST_MAX, 21).tolist()
NOISY_TEST_TRIALS = 20

# Plotting outputs
DIR_FIGS = os.path.join(DIR_PATH, 'figures')
F_FIG_GRID_PSBS = os.path.join(DIR_FIGS, 'grid_noisy_matrix.pdf')
F_FIG_FFT_PSBS = os.path.join(DIR_FIGS, 'fft_noisy_matrix.pdf')
F_FIG_COMPARE_GRID_FFT = os.path.join(DIR_FIGS, 'grid_vs_fft.pdf')
F_FIG_RAND_VS_ORD = os.path.join(DIR_FIGS, 'rand_vs_ordered.pdf')
F_FIG_FFT_DIFF_NH = os.path.join(DIR_FIGS, 'fft_diff_nh.pdf')

# Localized Noise
DIR_RESULTS = os.path.join(DIR_PATH, 'results')
DIR_LOC_NOISE = os.path.join(DIR_PATH, 'results', 'localized_noise')
F_LN_U_RAND = os.path.join(DIR_LOC_NOISE, 'U.npy')
F_LN_V_RAND = os.path.join(DIR_LOC_NOISE, 'V.npy')
F_LN_U_ORD = os.path.join(DIR_LOC_NOISE, 'U_ord.npy')
F_LN_V_ORD = os.path.join(DIR_LOC_NOISE, 'V_ord.npy')
F_LN_U_FFT = os.path.join(DIR_LOC_NOISE, 'U_fft.npy')
F_LN_V_FFT = os.path.join(DIR_LOC_NOISE, 'V_fft.npy')

F_FIG_LN_ORD = os.path.join(DIR_PATH, 'figures', 'loc_noise_ord.pdf')
F_FIG_LN_RAND = os.path.join(DIR_PATH, 'figures', 'loc_noise_rand.pdf')
F_FIG_LN_FFT = os.path.join(DIR_PATH, 'figures', 'loc_noise_fft.pdf')

# Noisy Prob
DIR_NOISY_PROB = os.path.join(DIR_RESULTS, 'noisy_prob')
