import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import complex_torch_var as ct
from numpy.linalg import svd as np_svd
from time import time
import matplotlib.pylab as plt
from unitary_decomp import unitary_decomp
from math import log2, ceil, pi, log, exp
from functools import partial

def svd(M, rand_S=True):
    """
    Performs SVD decomposition. optionally, randomly permutes the singular values.
    Args:
        M: A numpy array to be decomposed
        rand_S: If True, randomly permutes the singular values.

    Returns:
        U, VH: Unitary np.arrays
        S: 1-d np.array of singular values
    """
            
    U, S, VH =  np_svd(M)
    if not rand_S:
        return U, S, VH
    else:
        Di = VH.shape[0]
        Do = U.shape[0]
        if Di > Do:
            perm = np.random.permutation(Do)
            perm_ = np.arange(Di)
            perm_[:Do] = perm
            return U[:, perm], S[perm], VH[perm_, :]
        else:
            perm = np.random.permutation(Di)
            perm_ = np.arange(Do)
            perm_[:Di] = perm
            return U[:, perm_], S[perm], VH[perm, :]
def fft_idx(p, j):
    """
    Gives the indices for efficient implementation of FFTUnitary
        Args:
            p: The dimension of FFT layer is 2**p
            j: The layer for which indices are to be returned

        Returns:
            idx_uv : The inverse indices to be mixed. Used in defining UV_FFT
            idx_x : The indices used to permute X

        Example:
            fft_idx(3, 2) = 
                (tensor([0, 2, 4, 6, 1, 3, 5, 7]), tensor([4, 5, 6, 7, 0, 1, 2, 3]))

            The channels being mixed are (0, 4), (1, 5), (2, 6), and (3, 7)

            UV are constructed as

            UV <- UV[0, 2, 4, 6, 1, 3, 5, 7]
            or
            UV[0, 4, 1, 5, 2, 6, 3, 7] <- UV

            Pairs of channels mixed as desired

            The original input : 0 1 2 3 4 5 6 7
            Permuted input     : 4 5 6 7 0 1 2 3

            Again, the channels mixed are as expected
    """

    assert j < p
    perm = th.arange(2**p)
    _mask = 2**(j+1) - 1
    __mask = 2**j - 1
    mask_ = ~_mask
    perm_ = perm & mask_
    _perm = perm & _mask
    __perm = perm & __mask
    idx_uv = (__perm << 1) + (_perm >> j) + perm_
    idx_x = perm ^ 2**(j)

    return idx_uv, idx_x
def get_UV(theta, phi, dr=0, dr_=0):
    r = 2**(-0.5) + dr
    r_ = 2**(-0.5) + dr_

    t = (1 - r**2)**0.5
    t_ = (1 - r_**2)**0.5

    s_phi, c_phi = th.sin(phi), th.cos(phi)
    s_theta, c_theta = th.sin(theta), th.cos(theta)
    s_sum, c_sum = th.sin(theta + phi), th.cos(theta + phi)

    u_re = (r*r_*c_sum - t*t_*c_phi, r*r_ - t*t_*c_theta)
    v_re = (-r*t_*s_theta, -r_*t*s_sum - r*t_*s_phi)
    u_im = (r*r_*s_sum - t*t_*s_phi, -t*t_*s_theta)
    v_im = (t*r_ + r*t_*c_theta, t*r_*c_sum + r*t_*c_phi)

    return u_re, v_re, u_im, v_im
def UV_MZ(D, theta, phi, stage, BS_noise=[1, 1], new=False):

    n_layers = theta.shape[0]
    assert phi.shape[0] == n_layers

    if stage == 'A':
        idx_1 = slice(None, -1, 2)
        idx_2 = slice(1, None, 2)
    elif stage == 'B':
        idx_1 = slice(1, -1, 2)
        idx_2 = slice(2, None, 2)
    else:
        raise Exception('Incorrect stage name (A or B)')

    s1 = th.sin(theta/2)
    c1 = th.cos(theta/2)
    s2 = th.sin(theta/2 + phi)
    c2 = th.cos(theta/2 + phi) 

    device = th.device('cuda' if s1.is_cuda  else 'cpu')
    u_re = th.ones(n_layers, D).to(device)
    v_re = th.zeros(n_layers, D).to(device)
    u_im = th.zeros(n_layers, D).to(device)
    v_im = th.zeros(n_layers, D).to(device)

    noise_U, noise_V = BS_noise

    if new:
        ur, vr, ui, vi = get_UV(theta, phi, BS_noise[0], BS_noise[1])
        u_re[:, idx_1], u_re[:, idx_2] = ur
        v_re[:, idx_1], v_re[:, idx_2] = vr
        u_im[:, idx_1], u_im[:, idx_2] = ui
        v_im[:, idx_1], v_im[:, idx_2] = vi
    else:
        u_re[:, idx_1] = -s1*s2 * noise_U
        u_re[:, idx_2] = s1**2 * noise_U

        v_re[:, idx_1] = -c1*s1 * noise_V
        v_re[:, idx_2] = -c1*s2 * noise_V

        u_im[:, idx_1] = s1*c2 * noise_U
        u_im[:, idx_2] = -c1*s1 * noise_U

        v_im[:, idx_1] = c1**2 * noise_V
        v_im[:, idx_2] = c1*c2 * noise_V

    return u_re, v_re, u_im, v_im
def UV_FFT(theta, phi, BS_noise=[1, 1], new=False):
    """
        theta -- a (P, D//2) tensor : The internal phaseshifts
        phi -- (P, D//2) tensor : The external phaseshifts
    """

    n_layers = theta.shape[0]
    P, D2 = phi.shape
    assert P, D2 == theta.shape
    D = D2 * 2
    assert D == 2**P

    # Calculate the sin's and cos's
    s1 = th.sin(theta/2)
    c1 = th.cos(theta/2)
    s2 = th.sin(theta/2 + phi)
    c2 = th.cos(theta/2 + phi) 

    # Initialize UV
    device = th.device('cuda' if s1.is_cuda  else 'cpu')
    u_re = th.zeros(n_layers, D).to(device)
    v_re = th.zeros(n_layers, D).to(device)
    u_im = th.zeros(n_layers, D).to(device)
    v_im = th.zeros(n_layers, D).to(device)

    uv = [u_re, v_re, u_im, v_im]

    # Concat uv so that they're like u1u1 u2u2, ...
    idx_1 = slice(None, None, 2) #::2
    idx_2 = slice(1, None, 2) #1::2

    noise_U, noise_V = BS_noise

    if new:
        ur, vr, ui, vi = get_UV(theta, phi, BS_noise[0], BS_noise[1])
        u_re[:, idx_1], u_re[:, idx_2] = ur
        v_re[:, idx_1], v_re[:, idx_2] = vr
        u_im[:, idx_1], u_im[:, idx_2] = ui
        v_im[:, idx_1], v_im[:, idx_2] = vi
    else:
        u_re[:, idx_1] = -s1*s2 * noise_U
        u_re[:, idx_2] = s1**2 * noise_U
        v_re[:, idx_1] = -c1*s1 * noise_V
        v_re[:, idx_2] = -c1*s2 * noise_V
        u_im[:, idx_1] = s1*c2 * noise_U
        u_im[:, idx_2] = -c1*s1 * noise_U
        v_im[:, idx_1] = c1**2 * noise_V
        v_im[:, idx_2] = c1*c2 * noise_V


    # Put them in desired order
    for j in range(P):
        uv_idx, _ = fft_idx(P, j)
        u_re[j] = u_re[j, uv_idx] 
        v_re[j] = v_re[j, uv_idx] 
        u_im[j] = u_im[j, uv_idx] 
        v_im[j] = v_im[j, uv_idx] 

    return uv
def perm_full(D, stage='A', complex='True'):
    perm = list(range(D))
    if stage=='A':
        for i in range(D//2):
            perm[2*i], perm[2*i+1] = perm[2*i+1], perm[2*i]
    else:
        for i in range((D-1)//2):
            perm[2*i+1], perm[2*i+2] = perm[2*i+2], perm[2*i+1]

    return perm
def layer_mult_full(X, UV, perm):
    """
    Performs calculations equivalent to propgation through one layer of MZI.
    Args:
        X: (N, D)-th.Tensor representing the input with N being the batch size, D the dimension
        UV = [U_re, V_re, U_im, V_im]: The 1-D tensors containing the values of the transfer matrices of the MZI layer

    Returns:
        The output equivalent to U @ X where U would be the transfer matrix.
    """
        
    N, D2 = X.shape
    assert D2 % 2 == 0
    D = D2//2

    U_real, V_real, U_imag, V_imag = UV
    X_re = X[:, :D]
    X_im = X[:, D:]

    X_re = X[:, :D]
    X_im = X[:, D:]
    sX_re = X_re[:, perm]
    sX_im = X_im[:, perm]

    Y_real = (U_real*X_re - U_imag*X_im) + (V_real*sX_re - V_imag*sX_im)
    Y_imag = (U_real*X_im + U_imag*X_re) + (V_real*sX_im + V_imag*sX_re)

    return th.cat((Y_real, Y_imag), 1)


class FFTShaper(nn.Module):
    def __init__(self, D, dir, randomize=True):
        super().__init__()
        self.D = D
        self.D_pow_2 = 2**ceil(log2(D))
        assert dir in ['in', 'out']
        self.dir = dir

        perm = th.randperm(self.D_pow_2)
        self.idx = Parameter(perm[:D], requires_grad=False)
        #self.idx = th.arange(D)
    def forward(self, X):
        N, _ = X.shape
        device = th.device('cuda' if X.is_cuda  else 'cpu')
        if self.dir == 'in':
            # Complex variables
            out = th.zeros(N, self.D_pow_2 * 2).to(device)
            out[:, self.idx] = X[:, :self.D]
            out[:, self.idx + self.D_pow_2] = X[:, self.D:]
        else:
            out = th.zeros(N, self.D * 2).to(device)
            out[:, :self.D] = X[:, self.idx]
            out[:, self.D:] = X[:, self.idx + self.D_pow_2]
        return out

""" Unitary Modules """
class Unitary(nn.Module):
    """
    Custom pytorch Module simulating an ONN unitary multiplier

    Attributes:
        D: The dimension of multiplier
        sigma_PS: The stdev of gaussian noise added to phaseshifter values
        sigma_BS: The stdev of gaussian noise added to beamsplitter transmission
    """
    @classmethod
    def from_U(cls, U, numpy=False):
        """
        U : a complex unitary numpy matrix
        returns a onn Unitary with weights set to emulate U
        """
        if not numpy:
            D2 = U.shape[0]
            assert U.shape[1] == D2
            D = D2//2
            U_re = U[:D, :D].numpy()
            U_im = U[D:, :D].numpy()
            U = np.matrix(U_re + 1j*U_im)
        else:
            D = U.shape[0]
            assert U.shape[1] == D

        net = cls(D)
        for param, ang in zip(net.angles, unitary_decomp(U)):
            param.data = ang
        return net
    @classmethod
    def truncated(cls, n_layers):
        return partial(cls, n_layers=n_layers)
    def __init__(self, D, sigma_PS=0, sigma_BS=0, FFT=False, use_psi=True, n_layers=None, approx_sigma_bs=False):
        super().__init__()
        self.D = D
        if n_layers is None:
            self.n_layers = D
        else:
            self.n_layers = min(n_layers, D)
        self.use_psi = use_psi
        self.sigma_PS = sigma_PS
        self.sigma_BS = sigma_BS
        self.approx_sigma_bs = approx_sigma_bs
        self.init_params()
    def init_params(self):
        D = self.D
        n_layer_B = self.n_layers//2
        n_layer_A = self.n_layers - n_layer_B
        self.n_layers_A = n_layer_A
        self.n_layers_B = n_layer_B

        n_MZ_B = (D-1)//2
        n_MZ_A = D//2
        
        sin_A = th.rand(n_layer_A, n_MZ_A) 
        sin_B = th.rand(n_layer_B, n_MZ_B) 

        if False:
            Y_A = 2 * np.abs(np.arange(n_layer_A) * 2 - D/2) - 1
            Y_B = 2 * np.abs(np.arange(n_layer_B) * 2 - D/2) - 1
            X_A = 2 * np.abs(np.arange(n_MZ_A) * 2 - D/2) - 1
            X_B = 2 * np.abs(np.arange(n_MZ_B) * 2 - D/2) - 1
            
            XX_A, YY_A = np.meshgrid(X_A, Y_A)
            beta_A = D - np.maximum(XX_A, YY_A)
            
            alpha_A = np.ones_like(beta_A)
            sin_A = np.random.beta(alpha_A, beta_A)

            XX_B, YY_B = np.meshgrid(X_B, Y_B)
            beta_B = D - np.maximum(XX_B, YY_B)
            
            alpha_B = np.ones_like(beta_B)
            sin_B = np.random.beta(alpha_B, beta_B)

            sin_A = th.tensor(sin_A).float()
            sin_B = th.tensor(sin_B).float()

        self.phi_A = Parameter(th.rand(n_layer_A, n_MZ_A) * 1 * pi)
        self.phi_B = Parameter(th.rand(n_layer_B, n_MZ_B) * 1 * pi)
        self.theta_A = Parameter(th.asin(sin_A))
        self.theta_B = Parameter(th.asin(sin_B))

        # Phase shift at the end
        if self.use_psi:
            self.psi = Parameter(th.rand(D) * 2 * pi)
        else:
            self.psi = th.zeros(D)
    @property
    def angles(self):
        return [self.theta_A, self.phi_A, self.theta_B, self.phi_B, self.psi]
    def get_BS_noise(self):
        noise_UA = th.zeros_like(self.theta_A)
        noise_UB = th.zeros_like(self.theta_B)
        noise_VA = th.zeros_like(self.theta_A)
        noise_VB = th.zeros_like(self.theta_B)

        noise_UA.normal_()
        noise_UB.normal_()
        noise_VA.normal_()
        noise_VB.normal_()

        self.noise_A = [noise_UA, noise_VA]
        self.noise_B = [noise_UB, noise_VB]
        self.BS_noise_init = True
    def noisy_weights(self):
        """
        Add guassian noise of stdev sigma to all the angles
        """
        noisy_angles = []
        for angle in self.angles[:-1]:
            device = th.device('cuda' if angle.is_cuda else 'cpu')
            noise = th.zeros_like(angle).to(device)
            noise.normal_()
            noisy_angles.append(angle + self.sigma_PS * noise)
        if self.use_psi:
            noise = th.zeros_like(self.psi)
            noise.normal_()
            noisy_angles.append(self.psi + self.sigma_PS * noise)
        else:
            noisy_angles.append(self.psi)

        return noisy_angles
    def get_UV(self):
        # If simulating PS noise
        if self.sigma_PS > 0:
            theta_A, phi_A, theta_B, phi_B, psi = self.noisy_weights()
        else:
            theta_A, phi_A, theta_B, phi_B, psi = self.angles

        # If simulating BS noise
        if self.sigma_BS > 0:
            self.get_BS_noise()

            # If approximating BS noise
            if self.approx_sigma_bs:
                UV_A = UV_MZ(self.D, theta_A, phi_A, 'A')
                UV_B = UV_MZ(self.D, theta_B, phi_B, 'B')
                d_UV_A = UV_MZ(self.D, theta_A + pi, phi_A + pi, 'A', BS_noise=self.noise_A)
                d_UV_B = UV_MZ(self.D, theta_B + pi, phi_B + pi, 'B', BS_noise=self.noise_B)
                
                UV_A = [UV + 2**0.5 * self.sigma_BS * dUV for (UV, dUV) in zip(UV_A, d_UV_A)]
                UV_B = [UV + 2**0.5 * self.sigma_BS * dUV for (UV, dUV) in zip(UV_B, d_UV_B)]
            else:
                UV_A = UV_MZ(self.D, theta_A, phi_A, 'A', BS_noise=[self.sigma_BS * x for x in self.noise_A], new=True)
                UV_B = UV_MZ(self.D, theta_B, phi_B, 'B', BS_noise=[self.sigma_BS * x for x in self.noise_B], new=True)
        else:
            UV_A = UV_MZ(self.D, theta_A, phi_A, 'A')
            UV_B = UV_MZ(self.D, theta_B, phi_B, 'B')

        return UV_A, UV_B, psi
    def forward(self, X):
        UV_A, UV_B, psi = self.get_UV()
        perm_A = perm_full(self.D, 'A')
        perm_B = perm_full(self.D, 'B')

        # Iternate over the layers
        num_layers_total = UV_A[0].shape[0] + UV_B[0].shape[0]
        #for n in range(self.D):
        for n in range(num_layers_total):
            if n % 2 == 0:
                uv = [w[n//2] for w in UV_A]
                perm = perm_A
            else:
                uv = [w[(n-1)//2] for w in UV_B]
                perm = perm_B
            X = layer_mult_full(X, uv, perm)

        # Add final phase shift
        if self.use_psi:
            X_re = X[:, :self.D]
            X_im = X[:, self.D:]
            U_real = th.cos(psi)
            U_imag = th.sin(psi)
            Y_real = (U_real*X_re - U_imag*X_im)
            Y_imag = (U_real*X_im + U_imag*X_re)

            X = th.cat((Y_real, Y_imag), 1)

        return X
    def emul_U(self, U, numpy=False):
        if not numpy:
            D2 = U.shape[0]
            assert U.shape[1] == D2
            D = D2//2
            U_re = U[:D, :D].numpy()
            U_im = U[D:, :D].numpy()
            U = np.matrix(U_re + 1j*U_im)
        else:
            D = U.shape[0]
            assert U.shape[1] == D

        assert D == self.D

        for param, ang in zip(self.angles, unitary_decomp(U)):
            param.data = ang    
    def get_U(self, numpy=True):
        U = self(th.eye(self.D * 2)).data.t()
        U_re = U[:self.D, :self.D]
        U_im = U[self.D:, :self.D]
        if numpy:
            return np.matrix(U_re) + 1j * np.matrix(U_im)
        else:
            return U

class FFTUnitary(nn.Module):
    def __init__(self, D, sigma_PS=0, sigma_BS=0, use_psi=True, approx_sigma_bs=False):
        assert D & (D - 1) == 0 # Check if power of 2
        super().__init__()
        self.D = D
        self.P = int(log2(D))

        self.BS_noise_init = False

        self.use_psi = use_psi
        self.sigma_PS = sigma_PS
        self.sigma_BS = sigma_BS
        self.approx_sigma_bs = approx_sigma_bs
        self.init_params()
    def init_params(self):
        D = self.D
        P = self.P
        sin_theta = th.rand(P, D//2)
        self.theta = Parameter(th.asin(sin_theta))
        self.phi = Parameter(th.rand(P, D//2) * 2 * pi)

        # Phase shift at the end
        if self.use_psi:
            self.psi = Parameter(th.rand(D) * 2 * pi)
        else:
            self.psi = th.zeros(D)

        self.angles = [self.theta, self.phi, self.psi]
    def noisy_weights(self):
        """
        Add guassian noise of stdev sigma to all the angles
        """
        noisy_angles = []
        for angle in self.angles[:-1]:
            device = th.device('cuda' if angle.is_cuda else 'cpu')
            noise = th.zeros_like(angle).to(device)
            noise.normal_()
            noisy_angles.append(angle + self.sigma_PS * noise)
        if self.use_psi:
            noise = th.zeros_like(self.psi)
            noise.normal_()
            noisy_angles.append(self.psi + self.sigma_PS * noise)
        else:
            noisy_angles.append(self.psi)

        return noisy_angles
    def get_UV(self):
        # If simulating PS noise
        if self.sigma_PS > 0:
            theta, phi, psi = self.noisy_weights()
        else:
            theta, phi, psi = self.angles

        # If simulating BS noise
        if self.sigma_BS > 0:
            noise_U = th.zeros_like(self.theta)
            noise_V = th.zeros_like(self.theta)
            noise_U.normal_()
            noise_V.normal_()

            if self.approx_sigma_bs:
                UV = UV_FFT(self.theta, self.phi)
                d_UV = UV_FFT(self.theta + pi, self.phi + pi, BS_noise=[noise_U, noise_V])
                UV = [UV + 2**0.5 * self.sigma_BS * dUV for (UV, dUV) in zip(UV, d_UV)]
            else:
                UV = UV_FFT(self.theta, self.phi, BS_noise=[self.sigma_BS * noise_U, self.sigma_BS * noise_V], new=True)
        else:
            UV = UV_FFT(self.theta, self.phi)
        return UV, psi


    def forward(self, X):
        UV, psi = self.get_UV()
        # Iternate over the layers
        for n in range(self.P):
            uv = [x[n] for x in UV]
            _, perm = fft_idx(self.P, n)
            X = layer_mult_full(X, uv, perm)

        # Add final phase shift
        if self.use_psi:
            X_re = X[:, :self.D]
            X_im = X[:, self.D:]
            U_real = th.cos(psi)
            U_imag = th.sin(psi)
            Y_real = (U_real*X_re - U_imag*X_im)
            Y_imag = (U_real*X_im + U_imag*X_re)

            X = th.cat((Y_real, Y_imag), 1)
        return X
    def get_U(self, numpy=True, as_param=False):
        if as_param:
            U = self(th.eye(self.D * 2)).t()
        else:
            U = self(th.eye(self.D * 2)).data.t()
        U_re = U[:self.D, :self.D]
        U_im = U[self.D:, :self.D]
        if numpy:
            return np.matrix(U_re) + 1j * np.matrix(U_im)
        else:
            return U

class StackedFFTUnitary(nn.Sequential):
    def __init__(self, D, n_stack=None, sigma_PS=0, sigma_BS=0):
        if n_stack is None:
            P = int(log2(D))
            n_stack = int(D//P)
        layers = [FFTUnitary(D, sigma_PS=sigma_PS, sigma_BS=sigma_BS) for _ in range(n_stack)]

        super().__init__(*layers)
        self.sigma_PS = sigma_PS
        self.sigma_BS = sigma_BS
    @property
    def sigma_PS(self):
        return self._sigma_PS
    @property
    def sigma_BS(self):
        return self._sigma_BS
    @sigma_PS.setter
    def sigma_PS(self, new_sig):
        # Updates sigma of all layers
        for layer in self:
            layer.sigma_PS = new_sig
        self._sigma_PS = new_sig
    @sigma_BS.setter
    def sigma_BS(self, new_sig):
        # Updates sigma of all layers
        for layer in self:
            layer.sigma_BS = new_sig
        self._sigma_BS = new_sig

class HybridUnitary(Unitary):
    def forward(self, X):
        if self.sigma_PS > 0:
            theta_A, phi_A, theta_B, phi_B, psi = self.noisy_weights()
        else:
            theta_A, phi_A, theta_B, phi_B, psi = self.angles

        UV_A = UV_MZ(self.D, theta_A, phi_A, 'A')
        UV_B = UV_MZ(self.D, theta_B, phi_B, 'B')

        if self.sigma_BS > 0:
            d_UV_A = UV_MZ(self.D, theta_A + pi, phi_A + pi, 'A')
            d_UV_B = UV_MZ(self.D, theta_B + pi, phi_B + pi, 'B')

            if self.static_BS:
                if (self.noise_BS_B != None) and (self.noise_BS_A != None):
                    noise_A = self.noise_BS_A
                    noise_B = self.noise_BS_B
                else:
                    noise_A = th.zeros_like(d_UV_A[0])
                    noise_B = th.zeros_like(d_UV_B[0])
                    noise_A.normal_()
                    noise_B.normal_()
                    self.noise_BS_A = noise_A
                    self.noise_BS_B = noise_B
            else:
                noise_A = th.zeros_like(d_UV_A[0])
                noise_B = th.zeros_like(d_UV_B[0])
                noise_A.normal_()
                noise_B.normal_()

            
            UV_A = [UV + self.sigma_BS * noise_A * dUV for (UV, dUV) in zip(UV_A, d_UV_A)]
            UV_B = [UV + self.sigma_BS * noise_B * dUV for (UV, dUV) in zip(UV_B, d_UV_B)]
        
        perm_A = perm_full(self.D, 'A')
        perm_B = perm_full(self.D, 'B')

        # Iternate over the layers
        num_layers_total = UV_A[0].shape[0] + UV_B[0].shape[0]

        n_fft_perms = np.log2(self.D)
        n_layers_btw = int(num_layers_total // (n_fft_perms + 1))
        self.n_layers_btw = n_layers_btw
        for n in range(num_layers_total):
            if n % 2 == 0:
                uv = [w[n//2] for w in UV_A]
                perm = perm_A
            else:
                uv = [w[(n-1)//2] for w in UV_B]
                perm = perm_B
            X = layer_mult_full(X, uv, perm)

            if (n % n_layers_btw == 0) and (n != 0) and (n <= n_fft_perms * n_layers_btw):
                u = int(n // n_layers_btw) - 1
                idx = np.arange(X.shape[1])
                fft_perm = idx ^ 2**u
                X = X[:, fft_perm]



        # Add final phase shift
        if self.use_psi:
            X_re = X[:, :self.D]
            X_im = X[:, self.D:]
            U_real = th.cos(psi)
            U_imag = th.sin(psi)
            Y_real = (U_real*X_re - U_imag*X_im)
            Y_imag = (U_real*X_im + U_imag*X_re)

            X = th.cat((Y_real, Y_imag), 1)

        return X

class CGRDUnitary(Unitary):
    @staticmethod
    def get_perms(N):
        n_perm = int(np.ceil(np.log2(N)))
        perm_loc = (N / n_perm) * np.arange(n_perm)
        perm_depth = N ** (np.arange(n_perm)/n_perm)
        perm_depth[0] = 0

        # Ensure they are divisible by 2
        perm_loc = (perm_loc // 2 * 2).astype(int)
        perm_depth = (np.ceil(perm_depth/2) * 2).astype(int)

        # Define even and odd permutations
        perm_A =  np.arange(N)
        perm_B =  np.arange(N)
        N_2 = N//2 * 2
        perm_A[:N_2:2] += 1
        perm_A[1:N_2:2] -= 1
        perm_B[1:N_2-1:2] += 1
        perm_B[2:N_2-1:2] -= 1

        perms = []
        perm = np.arange(N)
        perm_idx = 0

        for i in range(N):
            if perm_idx == n_perm:
                break
            if i == perm_depth[perm_idx]:
                perms.append(perm.tolist())
                perm_idx += 1

            permutation = perm_A if i%2 ==0 else perm_B
            perm[permutation] = perm.copy()
        return perm_loc[1:], perms[1:]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        perm_loc, perms = self.get_perms(self.D)
        self.perm_dict = {}
        for loc, perm in zip(perm_loc, perms):
            self.perm_dict[loc] = perm + [x + self.D for x in perm]
    def forward(self, X):
        if self.sigma_PS > 0:
            theta_A, phi_A, theta_B, phi_B, psi = self.noisy_weights()
        else:
            theta_A, phi_A, theta_B, phi_B, psi = self.angles

        UV_A = UV_MZ(self.D, theta_A, phi_A, 'A')
        UV_B = UV_MZ(self.D, theta_B, phi_B, 'B')

        if self.sigma_BS > 0:
            d_UV_A = UV_MZ(self.D, theta_A + pi, phi_A + pi, 'A')
            d_UV_B = UV_MZ(self.D, theta_B + pi, phi_B + pi, 'B')

            if self.static_BS:
                if (self.noise_BS_B is not None) and (self.noise_BS_A is not None):
                    noise_A = self.noise_BS_A
                    noise_B = self.noise_BS_B
                else:
                    noise_A = th.zeros_like(d_UV_A[0])
                    noise_B = th.zeros_like(d_UV_B[0])
                    noise_A.normal_()
                    noise_B.normal_()
                    self.noise_BS_A = noise_A
                    self.noise_BS_B = noise_B
            else:
                noise_A = th.zeros_like(d_UV_A[0])
                noise_B = th.zeros_like(d_UV_B[0])
                noise_A.normal_()
                noise_B.normal_()

            
            UV_A = [UV + self.sigma_BS * noise_A * dUV for (UV, dUV) in zip(UV_A, d_UV_A)]
            UV_B = [UV + self.sigma_BS * noise_B * dUV for (UV, dUV) in zip(UV_B, d_UV_B)]
        
        perm_A = perm_full(self.D, 'A')
        perm_B = perm_full(self.D, 'B')

        # Iternate over the layers
        num_layers_total = UV_A[0].shape[0] + UV_B[0].shape[0]
        #for n in range(self.D):
        for n in range(num_layers_total):
            if n in self.perm_dict:
                idx = self.perm_dict[n]
                X = X[:, idx]
            if n % 2 == 0:
                uv = [w[n//2] for w in UV_A]
                perm = perm_A
            else:
                uv = [w[(n-1)//2] for w in UV_B]
                perm = perm_B
            X = layer_mult_full(X, uv, perm)

        # Add final phase shift
        if self.use_psi:
            X_re = X[:, :self.D]
            X_im = X[:, self.D:]
            U_real = th.cos(psi)
            U_imag = th.sin(psi)
            Y_real = (U_real*X_re - U_imag*X_im)
            Y_imag = (U_real*X_im + U_imag*X_re)

            X = th.cat((Y_real, Y_imag), 1)

        return X

class StackedFFTUnitary(nn.Sequential):
    def __init__(self, D, n_stack=1, sigma_PS=0, sigma_BS=0):
        layers = [FFTUnitary(D, sigma_PS=sigma_PS, sigma_BS=sigma_BS) for _ in range(n_stack)]
        self.D = D
        super().__init__(*layers)
        self.sigma_PS = sigma_PS
        self.sigma_BS = sigma_BS
    @property
    def sigma_PS(self):
        return self._sigma_PS
    @property
    def sigma_BS(self):
        return self._sigma_BS
    @sigma_PS.setter
    def sigma_PS(self, new_sig):
        # Updates sigma of all layers
        for layer in self:
            layer.sigma_PS = new_sig
        self._sigma_PS = new_sig
    @sigma_BS.setter
    def sigma_BS(self, new_sig):
        # Updates sigma of all layers
        for layer in self:
            layer.sigma_BS = new_sig
        self._sigma_BS = new_sig
    def get_U(self, numpy=True):
        U = self(th.eye(self.D * 2)).data.t()
        U_re = U[:self.D, :self.D]
        U_im = U[self.D:, :self.D]
        if numpy:
            return np.matrix(U_re) + 1j * np.matrix(U_im)
        else:
            return U

""" Diagonal Module """
class Diagonal(nn.Module):
    def __init__(self, D_in, D_out, sigma=0):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.D_min = min(D_in, D_out)
        self.sigma = sigma
        
        self.init_params()
    def init_params(self):
        sin_theta = th.rand(self.D_min) 

        self.theta = Parameter(th.asin(sin_theta))
        # Physically the amplification should happen at the nonlinearity but formally, it's cleaner here
        self.amp = Parameter(th.tensor(6.))
    def forward(self, X):
        N, D2 = X.shape
        assert D2 % 2 == 0
        D = D2//2
        assert D == self.D_in

        noise = th.zeros_like(self.theta)
        noise.normal_()
        theta = self.theta + self.sigma*noise

        u = th.sin(theta/2)
        device = th.device('cuda' if X.is_cuda else 'cpu')
        Y = th.zeros(N, 2 * self.D_out).to(device)
        Y[:, :self.D_min] = u * X[:, :self.D_min]
        Y[:, self.D_out:self.D_out+self.D_min] = u * X[:, D:D+self.D_min]
        return self.amp * Y

""" Linear Modules """
class Linear(nn.Module):
    @classmethod
    def from_M(cls, M, sigma_PS=0, sigma_BS=0, UNet=Unitary, numpy=False):
        if not numpy:
            Do2, Di2 = M.shape

            assert (Do2%2==0) and (Di2%2==0)
            Di = Di2//2
            Do = Do2//2
            M_re = M[:Do, :Di].numpy()
            M_im = M[Do:, :Di].numpy()
            M = np.matrix(M_re + 1j*M_im)
        else:
            Do, Di = M.shape

        net = cls(Di, Do, sigma_PS, sigma_BS, UNet)
        net.emul_M(M, numpy=True)
        return net
    def __init__(self, D_in, D_out, sigma_PS=0, sigma_BS=0, UNet=Unitary, FFT_shaper=False):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out

        # Define reshapers used for FFTNet
        self.in_shaper = self.out_shaper = None

        self.UNet = UNet

        if UNet == FFTUnitary:
            FFT_shaper = True
        if FFT_shaper:
            # Initialize the in/out shapers and obtain closest power of 2 dims
            D_in, D_out = self.init_fft()

        # SVD Decomp
        self.VH = UNet(D_in, sigma_PS=sigma_PS, sigma_BS=sigma_BS)
        self.S = Diagonal(D_in, D_out, sigma=sigma_PS)
        self.U = UNet(D_out, sigma_PS=sigma_PS, sigma_BS=sigma_BS)
        self.sigma_PS = sigma_PS
        self.sigma_BS = sigma_BS
    @property
    def sigma_PS(self):
        return self._sigma_PS
    @property
    def sigma_BS(self):
        return self._sigma_BS
    @sigma_PS.setter
    def sigma_PS(self, new_sig):
        # Updates sigma of all layers
        self.U.sigma_PS = new_sig
        self.VH.sigma_PS = new_sig
        self.S.sigma_PS = new_sig
        self._sigma_PS = new_sig
    @sigma_BS.setter
    def sigma_BS(self, new_sig):
        # Updates sigma of all layers
        self.U.sigma_BS = new_sig
        self.VH.sigma_BS = new_sig
        self.S.sigma_BS = new_sig
        self._sigma_BS = new_sig
    def init_fft(self):
        Di, Do = self.D_in, self.D_out
        if (Di - 1) & Di != 0:
            self.in_shaper = FFTShaper(Di, 'in')
            Di = self.in_shaper.D_pow_2
        if (Do - 1) & Do != 0:
            self.out_shaper = FFTShaper(Do, 'out')
            Do = self.out_shaper.D_pow_2
        return Di, Do
    def forward(self, X):
        #Make X a "complex vector" if not already
        N, D = X.shape
        if D == self.D_in:
            X = ct.make_batched_vec(X)
        else:
            assert D == self.D_in*2

        # Reshape input if needed for FFT
        if self.in_shaper is not None:
            X = self.in_shaper(X)

        X = self.VH(X)
        X = self.S(X)
        X = self.U(X)

        # Reshape output if needed for FFT
        if self.out_shaper is not None:
            X = self.out_shaper(X)
        return X
    def emul_M(self, M, numpy=False, rand_S=True):
        if self.UNet != Unitary:
            raise Exception('Decomposition of arbitrary matrices is only supported with GridUnitary.')
        if not numpy:
            Do2, Di2 = M.shape
            assert (Do2%2==0) and (Di2%2==0)
            Di = Di2//2
            Do = Do2//2
            M_re = M[:Do, :Di].numpy()
            M_im = M[Do:, :Di].numpy()
            M = np.matrix(M_re + 1j*M_im)
        else:
            Do, Di = M.shape

        U, S, VH = svd(M, rand_S=rand_S)

        S = th.tensor(S)
        amp = S.max()
        theta_diag = 2 * th.asin(S/amp)

        self.VH.emul_U(VH, True)
        self.U.emul_U(U, True)

        self.S = Diagonal(Di, Do)
        self.S.theta.data = theta_diag
        self.S.amp.data = amp
    def get_M(self, numpy=True):
        U = self(th.eye(self.D_in * 2)).data.t()
        U_re = U[:self.D_out, :self.D_in]
        U_im = U[self.D_out:, :self.D_in]
        if numpy:
            return np.matrix(U_re) + 1j * np.matrix(U_im)
        else:
            return U

class TruncatedGridLinear(Linear):
    def __init__(self, D_in, D_out, sigma_PS=0, sigma_BS=0):
        super().__init__(D_in, D_out, sigma_PS, sigma_BS, UNet=Unitary)

        # Initialize the in/out shapers and obtain closest power of 2 dims
        D_in, D_out = self.init_fft()
        P_in = int(log2(D_in))
        P_out = int(log2(D_out))

        U_in = Unitary.truncated(P_in)
        U_out = Unitary.truncated(P_out)

        # SVD Decomp
        self.VH = U_in(D_in, sigma_PS=sigma_PS, sigma_BS=sigma_BS)
        self.S = Diagonal(D_in, D_out, sigma=sigma_PS)
        self.U = U_out(D_out, sigma_PS=sigma_PS, sigma_BS=sigma_BS)
        self.sigma_PS = sigma_PS
        self.sigma_BS = sigma_BS

class ComplexLinear(nn.Module):
    def __init__(self, D_in, D_out, sigma=0, has_bias=False):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.has_bias = has_bias
        self.init_params()
        self.sigma = sigma
    def init_params(self):
        U = ct.rand_unitary(self.D_out)
        S = th.zeros(self.D_out, self.D_in)
        VH = ct.rand_unitary(self.D_in)

        sigma = 1/(self.D_in + self.D_out) ** 0.5
        sigma *= 20
        if self.D_out < self.D_in:
            diag = th.randn(self.D_out) * sigma
            S[:, :self.D_out] = th.diag(diag)
        else:
            diag = th.randn(self.D_in) * sigma
            S[:self.D_in, :] = th.diag(diag)
        S = ct.make_complex_matrix(S)
        
        M = (U@S@VH)
        self.M_real = Parameter(M[:self.D_out, :self.D_in])
        self.M_imag = Parameter(M[self.D_out:, :self.D_in])

        if self.has_bias:
            self.bias = Parameter(th.Tensor(D_out*2))
            self.bias.data.uniform_(-sigma, sigma)
        else:
            self.register_parameter('bias', None)
    @property
    def weight(self):
        return ct.make_complex_matrix(self.M_real, self.M_imag)
    def set_weight(self, M):
        self.M_real.data = M[:self.D_out, :self.D_in]
        self.M_imag.data = M[self.D_out:, :self.D_in]
    def forward(self, X):
        if self.sigma > 0:
            device = th.device('cuda' if self.weight.is_cuda else 'cpu')
            noise = th.zeros_like(self.weight).to(device)
            noise.normal_()
            weight = self.weight + noise * self.sigma
        else:
            weight = self.weight

        return F.linear(X, weight, self.bias)
    def get_M(self, numpy=True):
        U = self(th.eye(self.D_in * 2)).data.t()
        U_re = U[:self.D_out, :self.D_in]
        U_im = U[self.D_out:, :self.D_in]
        if numpy:
            return np.matrix(U_re) + 1j * np.matrix(U_im)
        else:
            return U

""" Nonlinearities """
class ModNonlinearity(nn.Module):
    def __init__(self, f=None):
        """
        Impliments nonlinearity that acts on the magnitude of a complex vector, leaving the phase the same.

        f : the nonlinearity to be used. Should be from torch.nn.functional for backprop to work
        """
        if f is None:
            f = ShiftedSoftplus(0.1)

        super().__init__()
        self.f = f
    def forward(self, Z):
        _, D = Z.shape
        # Z should be already a complex vector
        assert D % 2 == 0

        X = Z[:, :D//2]
        Y = Z[:, D//2:]

        Z_abs = ct.norm_squared(Z)**0.5
        W = self.f(Z_abs)
        U = X * W / Z_abs
        V = Y * W / Z_abs

        out = ct.make_batched_vec(U, V)
        return out

class ComplexNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return ct.norm_squared(X)

class ShiftedSoftplus(nn.Module):
    def __init__(self, T):
        super().__init__()
        # Calculate bias
        self.u0 = 0.5 * log(T**(-1) - 1)
    def forward(self, X):
        return 0.5 * (F.softplus(2 * (X - self.u0)) - log(1 + exp(-2*self.u0)))

""" Other """
class NoisySequential(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)
    def set_noise(self, PS, BS):
        for l in self:
            if isinstance(l, Linear):
                l.sigma_BS = BS
                l.sigma_PS = PS

if __name__ == '__main__':
    D = 4
    X = th.randn(1, D * 2)
    net = Unitary(D, sigma_BS=1e-2, approx_sigma_bs=False)
    U = net.get_U()
    print(U @ U.H)
