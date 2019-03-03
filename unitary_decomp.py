import numpy as np
from numpy.linalg import svd
from scipy.stats import unitary_group
import math
from math import pi
from complex_torch_var import *
import logging
from time import time
#from optical_nn_2 import DiagLayer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

def random_complex_mat(N, M=None):
    if M == None:
        M = N
    U = np.random.randn(N, M)
    V = np.random.randn(N, M)
    return np.matrix(U + 1j * V)
def rand_unitary(N):
    U = unitary_group.rvs(N)
    return np.matrix(U)
def rand_vec(D=2, complex=True):
    X = np.random.randn(D)
    Y = np.random.randn(D)
    if complex:
        return X + 1j * Y
    else:
        return X
def rand_theta_phi():
    theta, phi = np.random.uniform(0, np.pi, size=2)
    return theta, 2 * phi

def U_BS(alpha=np.pi/2):
    c = math.cos(alpha/2)
    s = math.sin(alpha/2)
    return np.matrix(
            [[c, 1j * s],
            [1j * s, c]]
            )
def U_PS(theta):
    phase = np.exp(1j * np.array([theta, 0]))
    return np.matrix(np.diag(phase))
def U_MZ(theta, phi, a1=np.pi/2, a2=np.pi/2):
    if (a1 == np.pi/2) and (a2 == np.pi/2):
        M = np.matrix(
                [
                    [np.exp(1j * phi) * np.sin(theta/2), np.cos(theta/2)],
                    [np.exp(1j * phi) * np.cos(theta/2), -np.sin(theta/2)]
                    ]
                )
        return 1j * np.exp(1j * theta/2) * M

    # If note 50-50 multiply through the components
    return U_BS(a1) @ U_PS(theta) @ U_BS(a2) @ U_PS(phi)
def get_null_angle(X, flip=False):
    """
    Given a complex 2-vector X, find angles theta, phi such that
    U_MZ(theta, phi) @ X = [1, 0] (or [0, 1] when flip == True)
    """

    if isinstance(X, np.matrix):
        X = np.array(X).flatten()
    X_mag = np.absolute(X)
    phis = np.angle(X)
    theta = np.arctan(X_mag[0] / X_mag[1]) * 2
    phi = (phis[1] - phis[0])

    if flip == True:
        theta = (np.pi - theta) 
        phi += np.pi


    phi %= 2 * math.pi
    return theta, phi
def get_sub_T(N, i, j, T):
    M = np.eye(N, dtype='complex128')
    M[i, i] = T[0, 0]
    M[i, j] = T[0, 1]
    M[j, i] = T[1, 0]
    M[j, j] = T[1, 1]

    return np.matrix(M)
def get_ij(N, s, k, backward=False):
    """
    Returns indices (i, j) of U that the kth step of stage s nulls
    """
    if backward:
        i = N - (s + 1 - k)
        j = k
    else:
        i = N - 1 - k
        j = s - k
    return i , j
def get_nl(D, s, k):
    """
    Returns the physical locations of the MZI associated with the angles obtained in stage s, step k
        n: The MZI swaps channels (n, n+1)
        l: The MZI is in layer l
    """

    n_max = D - 2
    l_max = D - 1

    n = s - k
    l = k
    if s % 2 == 1:
        n = n_max - n
        l = l_max - l
    return n, l
def null_element(U, i, j, backward=False):
    log.info(f'Nulling element U[{i}, {j}], {"backward" if backward else "forward"} stage')
    N, _ = U.shape
    if backward:
        X = U[[i-1, i], j]
        theta, phi = get_null_angle(X, flip=False)
        T = U_MZ(theta, phi)
        U[[i-1, i], :] = T @ U[[i-1, i], :]
    else:
        X = U[i, [j, j+1]]
        theta, phi = get_null_angle(X.H, flip=True)
        T = U_MZ(theta, phi)
        U[:, [j, j+1]] = U[:, [j, j+1]] @ T.H
    epsilon = 1e-8

    if np.abs(U[i, j]) > epsilon:
        log.warning(f'The element U[{i}, {j}] was not nulled within tolerance of {epsilon}, its abs value is {np.abs(U[i, j]):.2e}')
    else:
        log.info('Element sucessfully nulled')
    return (theta, phi)
    """
    Decomposes an unitary matrix to a series of SU(2) implimented by MZIs as described
    by Clements et. al (2017)
    
    U: the unitary matrix to be decopsed
    
    returns
        coords : coordinates of the MZIs
        angles : a (N (N - 1) / 2) x 2 array of thetas, phis which parametrize the MZIs
        D : a N vector which gives the residual phase shifts
    """
    # Makes a copy of U to return to as all operations are in place.
    if reset_U:
        U0 = U.copy()
    N, _ = U.shape
    n_stages = N - 1

    angles_f = []
    angles_b = []
    coords_f = []
    coords_b = []
    MZI_loc_f = []
    MZI_loc_b = []

    # Build the coordinates of operation
    for s in range(n_stages):
        # Odd iterations are backward stages
        backward_stage = (s % 2 == 1)
        for k in range(s + 1):
            n, l = get_nl(N, s, k)
            i, j = get_ij(N, s, k, backward=backward_stage)
            theta_phi = null_element(U, i, j, backward=backward_stage)
            if backward_stage:
                coords_b.append((i-1, i))
                angles_b.append(theta_phi)
                MZI_loc_b.append((l, n))
            else:
                coords_f.append((j, j+1))
                angles_f.append(theta_phi)
                MZI_loc_f.append((l, n))

    # Reverse order of backward operations
    angles_b.reverse()
    coords_b.reverse() 
    MZI_loc_b.reverse()

    # Put forward and backwards together
    coords = coords_f + coords_b
    MZI_loc = MZI_loc_f + MZI_loc_b

    def swap_T_D(theta_phi, D):
        # Find T_ and D_ such that T.H @ D = D_ @ T_
        theta, phi = theta_phi

        # Get phases of D
        psis = np.array(np.angle(D.diagonal()))[0]
        psi0 = psis[1]
        psi = psis[0] - psi0

        # Get new angles
        theta_ = theta
        phi_ = psi
        psi_ = -phi
        psi0_ = psi0 - theta + np.pi

        # Make new D
        D_ = np.exp(1j * psi0_) * U_PS(psi_)
        T_ = U_MZ(theta_, phi_)

        return (theta_, phi_), D_

    # Put angles in layers
    n_back = len(angles_b)
    for n in range(n_back):
        theta, phi = angles_b[n]
        i, j = coords_b[n]
        D = U[[i, j]][:, [i, j]]
        (theta_, phi_), D_ = swap_T_D((theta, phi), D)
        angles_b[n] = (theta_, phi_)
        #T_ = U_MZ(theta_, phi_)
        #sub_T_ = get_sub_T(N, i, j, T_)
        U[i, i] = D_[0, 0]
        U[j, j] = D_[1, 1]


    return angles_f
    # Put angles together and set to be in (0, 2pi)
    angles = angles_f + angles_b
    D = np.angle(np.diag(U)) % (2 * np.pi)

    # Brings U back to original input
    if reset_U:
        U = U0

    layered_angles = [[None,] * ((N)//2) for _ in range(N)]

    for angle, (l, n) in zip(angles, MZI_loc):
        if l % 2 == 0:
            i = n//2
        else:
            i = (n-1)//2
        layered_angles[l][i] = angle

    return layered_angles
    return coords, np.array(angles), D
def unitary_decomp(U, in_place=False):
    if U.dtype != 'complex128':
        U = U.astype('complex128')
    """
    Decomposes an unitary matrix to a series of SU(2) implimented by MZIs as described
    by Clements et. al (2017)
    
    U: the unitary matrix to be decopsed
    
    returns
        theta/phi_A/B : angles of the phase shifters at each layer
        psi : residual phase shift
    """
    if not in_place:
        U = U.copy()

    N, _ = U.shape
    n_stages = N - 1

    angles_f = []
    angles_b = []
    coords_f = []
    coords_b = []
    MZI_loc_f = []
    MZI_loc_b = []

    # Build the coordinates of operation
    for s in range(n_stages):
        # Odd iterations are backward stages
        backward_stage = (s % 2 == 1)
        for k in range(s + 1):
            n, l = get_nl(N, s, k)
            i, j = get_ij(N, s, k, backward=backward_stage)
            theta_phi = null_element(U, i, j, backward=backward_stage)
            if backward_stage:
                coords_b.append((i-1, i))
                angles_b.append(theta_phi)
                MZI_loc_b.append((l, n))
            else:
                coords_f.append((j, j+1))
                angles_f.append(theta_phi)
                MZI_loc_f.append((l, n))

    # Reverse order of backward operations
    angles_b.reverse()
    coords_b.reverse() 
    MZI_loc_b.reverse()


    # Put forward and backwards together
    coords = coords_f + coords_b
    MZI_loc = MZI_loc_f + MZI_loc_b

    def swap_T_D(theta_phi, D):
        # Find T_ and D_ such that T.H @ D = D_ @ T_
        theta, phi = theta_phi

        # Get residual phases
        psis = np.array(np.angle(D.diagonal()))[0]
        psi0 = psis[1]
        psi = psis[0] - psi0

        # Get new angles
        theta_ = theta
        phi_ = psi
        psi_ = -phi
        psi0_ = psi0 - theta + np.pi

        # Make new D
        D_ = np.exp(1j * psi0_) * U_PS(psi_)
        T_ = U_MZ(theta_, phi_)

        return (theta_, phi_), D_

    # Put angles in layers
    n_back = len(angles_b)
    for n in range(n_back):
        theta, phi = angles_b[n]
        i, j = coords_b[n]
        D = U[[i, j]][:, [i, j]]
        (theta_, phi_), D_ = swap_T_D((theta, phi), D)
        angles_b[n] = (theta_, phi_)
        #T_ = U_MZ(theta_, phi_)
        #sub_T_ = get_sub_T(N, i, j, T_)
        U[i, i] = D_[0, 0]
        U[j, j] = D_[1, 1]


    # Put angles together and set to be in (0, 2pi)
    angles = angles_f + angles_b
    psi = np.angle(np.diag(U)) % (2 * np.pi)

    # Initialize theta/phi_A/B
    n_layer_B = N//2
    n_layer_A = N - n_layer_B
    n_MZ_B = (N-1)//2
    n_MZ_A = N//2
    theta_A = th.zeros(n_layer_A, n_MZ_A) * 2 * pi
    phi_A = th.zeros(n_layer_A, n_MZ_A) * 2 * pi
    theta_B = th.zeros(n_layer_B, n_MZ_B) * 2 * pi
    phi_B = th.zeros(n_layer_B, n_MZ_B) * 2 * pi
    theta_phis = [theta_A, phi_A, theta_B, phi_B]
    for n in range(4):
        theta_phis[n] = theta_phis[n].float()

    for angle, (l, n) in zip(angles, MZI_loc):
        if l % 2 == 0:
            i = n//2
            theta_A[l//2, i] = angle[0]
            phi_A[l//2, i] = angle[1]
        else:
            i = (n-1)//2
            theta_B[(l-1)//2, i] = angle[0]
            phi_B[(l-1)//2, i] = angle[1]

    return theta_A, phi_A, theta_B, phi_B, th.tensor(psi).float()
def diag_decomp(S):
    """
    Given the diagonal of a non-negative diagonal matrix S (as a vector), find the angles (theta, phi) that impliments attenuation. Note that S will be normalized first so that the largest value will be 1 and all others less than 1.
    """
    # Normalize S
    scale = S.max()
    S_ = S/scale
    thetas = 2 * np.arcsin(S_)
    phis = -thetas/2 - np.pi/2
    return np.vstack((thetas, phis)).T, float(scale)

if __name__ == '__main__':
    D = 4
    U_im = np.eye(D)[::-1]
    U = np.matrix(-1j * U_im)
    print(type(U))
    for x in unitary_decomp(U):
        print(x/np.pi)
