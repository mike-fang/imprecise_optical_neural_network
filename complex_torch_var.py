import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from scipy.stats import unitary_group

def rand_unitary(N):
    """
    Returns a N x N randomly sampled unitary matrix represented as th.tensor
    """
    U = unitary_group.rvs(N)
    U_real = th.tensor(U.real)
    U_imag = th.tensor(U.imag)
    return make_complex_matrix(U_real, U_imag)
def make_complex_matrix(real, imag=None):
    """
    Creates a real representation of complex matrix of the form
        (real, -imag)
        (imag, real)
    Input:
        real: (N x M) th.tensor, real component of matrix
        imag: (N x M) th.tensor, imag component of matrix. If none given, assumed to be zeros.
    Returns:
        A (2N x 2M) th.tensor, a real representation of complex matrix
    """
            
    if imag is None:
        imag = real * 0
    # 2D matrix
    assert len(real.shape) == 2
    Z_upper = th.cat((real, -imag), dim=1)
    Z_lower = th.cat((imag, real), dim=1)
    Z = th.cat((Z_upper, Z_lower), dim=0)
    return Z.float()
def make_batched_vec(real, imag=None):
    """
    Represent complex input of shape (N, D) as real th.tensor. N is the batch size and D is the dimension
    """
    if imag is None:
        imag = real * 0
    # 2D batched vectors
    assert len(real.shape) == 2
    Z = th.cat((real, imag), dim=1)
    return Z
def norm_squared(Z):
    N, D_2 = Z.shape
    #assert D_2 % 2 == 0
    D = D_2//2
    real = Z[:, :D]
    imag = Z[:, D:]
    return (real**2 + imag**2)
def print_complex_mat(Z, prec=None, **kwarg):
    np.set_printoptions(precision=prec, **kwarg)
    N, M = Z.shape
    assert (N % 2 == 0) and (M % 2 == 0)
    N = N//2
    M = M//2
    real = Z[:N, :M]
    imag = -Z[:N, M:]
    if (isinstance(Z, np.ndarray)):
        Z = real + imag * 1j
    else:
        Z = real.data.numpy() + imag.data.numpy() * 1j
    print("Complex Tensor: \n" + str(Z))
def print_complex_vec(Z):
    _, N = Z.shape
    assert (N % 2 == 0)
    N = N//2
    real = Z[:, :N]
    imag = -Z[:, N:]

    if (isinstance(Z, th.Tensor)):
        Z = real.numpy() + imag.numpy() * 1j
    else:
        Z = real.data.numpy() + imag.data.numpy() * 1j
    print("Complex Tensor: \n" + str(Z))
def complex_torch_to_numpy(X):
    X = X.data.numpy()
    
    N, M = X.shape
    assert (N % 2 == 0) and (M % 2 ==0)
    real = X[:N//2, :M//2]
    imag = X[N//2:, :M//2]
    return np.matrix(real + 1j * imag)

class ComplexNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return norm_squared(X)

class ComplexLinear(nn.Module):
    def __init__(self, D_in, D_out, sigma=0, has_bias=False):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.has_bias = has_bias
        self.init_params()
        self.sigma = sigma
    def init_params(self):
        U = rand_unitary(self.D_out)
        S = th.zeros(self.D_out, self.D_in)
        VH = rand_unitary(self.D_in)

        sigma = 1/(self.D_in + self.D_out) ** 0.5
        if self.D_out < self.D_in:
            diag = th.randn(self.D_out) * sigma
            S[:, :self.D_out] = th.diag(diag)
        else:
            diag = th.randn(self.D_in) * sigma
            S[:self.D_in, :] = th.diag(diag)
        S = make_complex_matrix(S)
        
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
        return make_complex_matrix(self.M_real, self.M_imag)
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


if __name__ == '__main__': 
    D_in, D_out = 3, 4
    net = ComplexLinear(D_in, D_out, 1e-2)
    X = th.randn(1, D_in)
    X = make_batched_vec(X)
    print(net(X))
    print(net(X))
    print(net(X))
    print(net(X))
