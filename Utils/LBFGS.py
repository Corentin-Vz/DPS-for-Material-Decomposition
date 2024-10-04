import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import astra
import torch
import itertools

device = 'cuda'



def create_bins_and_mu():
    """Create energy bins and mass attenuation matrix as per shared model.

    :return: mu, bins
    """
    def create_bins(spectral_incident, bin_list):
        '''
        Spectral_incident : spectral incident intensity array
        bin_list : list of energy intervals
        returns h_k(E) function where k is the k-th bin.
        '''
        h_k = torch.zeros(len(bin_list) - 1, spectral_incident.shape[0], device=device)
        for k in range(len(bin_list) - 1):
            bin_k = torch.zeros(spectral_incident.shape[-1], device=device)
            bin_k[bin_list[k]:bin_list[k + 1]] = 1
            h_k[k, :] = spectral_incident * bin_k
        return h_k


    df = pd.read_csv('csv_files/spectral_incident.csv')
    S = torch.zeros(150)
    S = torch.tensor(df['120_keV'], device=device)

    bin_list = [10, 40, 60, 120]
    prop_factor = 0.1
    binned_spectrum = create_bins(prop_factor * S, bin_list)
    n_bin = len(bin_list) - 1
    S_k = torch.sum(binned_spectrum, dim=1)  # Total photon number per bin
    E_k_list = (torch.sum(torch.arange(150, device=device) * binned_spectrum, dim=1) / S_k)  # Mean energy per bin

    # Loading material composition matrix for every e in {1, 2, ..., 150keV}
    # ! Unit is cm^-1 !
    df = pd.read_csv('csv_files/mass_att.csv')  # mm^-1
    rho_df = pd.read_csv('csv_files/rho.csv')  # (not used : mass_att.csv already multiplied by rho)
    M = torch.zeros(150, 2, device=device)
    M[:, 1] = torch.tensor(df['Soft Tissues']) * 10
    M[:, 0] = torch.tensor(df['Bones']) * 10
    # Multiplication by 10 : from mm^-1 to cm^-1
    M[None, :, :] = M
    # Selecting average energy for each bin in order to create a pseudo spectral material composition matrix :
    M_pseudo_spectral = M[torch.round(E_k_list).long(), :]
    return M.detach().cpu().numpy(), binned_spectrum.detach().cpu().numpy()



@dataclass
class ProblemSetup:
    """Dataclass to hold information concerning imaging setup and measurement to optimize

    """
    y: np.array  # measurement
    h: np.array  # discretized forward spectrum support
    proj: astra.optomo.OpTomo  # forward projector
    img_shape: tuple
    sino_shape: tuple  # (n_angles, det_count)
    energies: np.array  # array of energies used
    n_mats: int
    mass_attn: np.array  # mass attenuation matrix
    y_bar_func: callable  # forward model


def grad_verification(f, xshape):
    """Verify analytical gradient via finite differences (FD) method.

    :param f: callable: function to verify; needs to return f(x) and grad f(x)
    :param xshape: shape of x to supply to f
    :return:
    """
    epsilon = .05  # step size for FD
    samples = 10  # num_samples. Note that every sample will be analyzed in every direction, resulting in samples*prod(xshape) function calls
    ana = np.zeros((samples, np.prod(xshape)))
    fd = np.zeros((samples, np.prod(xshape)))
    for i in range(len(ana)):
        x = np.random.normal(np.ones(xshape))
        x[x<0]=0
        fx, gfx = f(x)
        ana[i] = gfx  # analytical gradient at x
        for j in range(len(ana[i])):
            eps = np.zeros(ana[i].shape)
            eps[j]=1  # unit vector in direction j
            fx2, gfx2 = f(x+(epsilon*eps).reshape(xshape))
            fd[i,j] = (fx2-fx)/epsilon  #finite difference gradient
        plt.scatter(ana[i].flatten(), fd[i].flatten())
    plt.xlabel('ana')
    plt.ylabel('fd')
    #min_ = int(np.min([np.min(ana), np.min(fd)]))
    #max_ = int(np.max([np.max(ana), np.max(fd)]))
    #plt.plot(np.linspace(min_,max_,10),np.linspace(min_,max_,10))
    plt.show()  # if correct, y=x on this plot


def inner_prod_reg(im, weights=None):
    """Inner product regularizer. Supports arbitrary amount of materials, returns sum of pair-wise inner product.

    :param im: Input image. Inner product will be calculated in dimension 0.
    :param weights: np.array, shape (len(im)). Allows weighting of different channels. Equal weighting by default.
    :return: f(im), grad f(im)
    """
    if not weights:
        weights = np.ones(len(im))
    mat_img_indices = [x for x in range(len(im))]
    pairs = itertools.combinations(mat_img_indices, 2)
    f = 0
    g = np.zeros(im.shape)
    for pair in pairs:
        f += np.sum(im[pair[0]]*weights[pair[0]]*im[pair[1]]*weights[pair[1]])
        g[pair[0]] += im[pair[1]]*weights[pair[1]]
        g[pair[1]] += im[pair[0]]*weights[pair[0]]
    return f, g.flatten()


def huber(x, delta):
    """Huber regularization function.

    :param x: input image
    :param delta: Huber parameter
    :return:
    """
    psi0 = (.5*(x**2)) * (abs(x) <= delta) + (delta*abs(x) - (delta**2)/2)*(abs(x) > delta)
    psi1 = x*(abs(x) <= delta) + delta*(x > delta) - delta*(x < -delta)
    return psi0, psi1


def imagePrior(im, delta):
    """Calculates f, grad f with f the Huber prior.

    :param im: input image
    :param delta: huber parameter
    :return: f(im), grad f(im)
    """
    vect = []
    dir_list = [0, 1, -1]
    F = np.zeros(im.shape)
    D = np.zeros(im.shape)
    for idx, x in enumerate(dir_list):
        for idy, y in enumerate(dir_list):
            vect.append((x,y))
    for idm,mat_im in enumerate(im):
        working_D = np.zeros(mat_im.shape)
        for shift_vect in vect[1:]:
            im_shifted = np.roll(mat_im, shift_vect, axis=(0, 1))
            omega = np.ones(mat_im.shape)/np.linalg.norm(shift_vect)
            psi0, psi1 = huber(mat_im-im_shifted, delta)
            F += omega * psi0
            working_D += 2 * omega * psi1
        D[idm] = working_D.copy()
    return np.sum(F), D.flatten()


def forward(x, bins, n_angles, det_count, n_mats, image_shape, forward_op, mu):
    """ Calculate y_bar(x)

    :param x: np.array: input image
    :param bins: np.array: (n_bins, n_energies) photon counts of all energies for each bin
    :param n_angles:
    :param det_count:
    :param n_mats: number of materials
    :param image_shape:
    :param forward_op: forward projection operator
    :param mu: mass attenuation matrix
    :return:
    """
    proj = np.zeros((len(bins), n_angles, det_count))
    temp_ = mu @ x.reshape((n_mats, np.prod(image_shape)))

    temp_proj_ = np.zeros((len(temp_), n_angles, det_count))
    for E in range(len(temp_)):
        temp_proj_[E] = forward_op.FP(temp_[E])
    temp_proj_ = np.exp(-temp_proj_)
    for idh, h in enumerate(bins):
        proj[idh] = np.sum(h.reshape((len(h),1,1))*temp_proj_, axis=0)
    return proj


def fullcost(x, P, beta, gamma, delta):
    """Calculates full target function (likelihood + regularizers).

    :param x: input image
    :param P: Problem setup for likelihood calculation
    :param beta: huber prior weight
    :param gamma: inner product weight
    :param delta: huber prior parameter
    :return:
    """
    fr, gr = imagePrior(x, delta)
    fi, gi = inner_prod_reg(x, np.ones(len(x)))
    fl, gl = likelihood(x, P)
    ftot = fl + beta*fr + gamma*fi
    gtot = gl + beta*gr + gamma*gi
    return ftot, gtot


def likelihood(x: np.array, P: ProblemSetup):
    """Calculate negative log likelihood and its gradient

    :param x: np.array: image of shape (#materials, *image_shape)
    :param P: ProblemSetup: see ProblemSetup dataclass
    :return: L(x), grad L(x)
    """
    x = x.reshape((P.n_mats, np.prod(P.img_shape)))  # Flatten image dimension for LBFGS
    y_bar_x = P.y_bar_func(x)
    L = np.sum(-P.y * np.log(y_bar_x) + y_bar_x)

    grad_L = np.zeros((P.n_mats, np.prod(P.img_shape)))

    grad_phi = (-P.y / y_bar_x + 1)

    combined_mats = P.mass_attn @ x  # Combine materials
    exponent = np.zeros((len(P.energies), *P.sino_shape))
    for E in range(len(P.energies)):
        exponent[E] = P.proj.FP(combined_mats[E])  # Forward project the image at all energies
    att_term = np.exp(-exponent)
    for idm in range(P.n_mats):
        mul = P.mass_attn[:,idm].reshape((len(P.energies),1,1))  # reshape for broadcasting to work
        inner = np.zeros((len(P.h), *P.sino_shape))
        backprojected = np.zeros((len(P.h), *P.img_shape))
        for idh,h in enumerate(P.h):
            inner[idh] = np.sum(mul*h.reshape((len(h), 1, 1))*att_term,axis=0)  # sum over energies; this is the integral over E
            backprojected[idh] = P.proj.BP(inner[idh]*grad_phi[idh])  # backproject integral * phi'

        grad_L[idm] = np.sum(-backprojected, axis=0).flatten()  # sum over bins

    return L, grad_L.flatten()


def create_forward_op(img_shape):
    """Create astra forward operator.

    :param img_shape: (int,int)
    :return:
    """
    n_angles = 180
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    pix_size = 1 / 2
    det_width = 1.2 / pix_size
    det_count = 75
    sino_shape = (n_angles, det_count)
    source_origin = 600 / pix_size
    origin_det = 600 / pix_size
    vol_geom = astra.create_vol_geom(*img_shape)
    proj_geom = astra.create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det)
    proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
    forward_op = astra.OpTomo(proj_id)
    return n_angles, det_count, forward_op


def LBFGS(x_init, sino):
    """Run the LBFGS code on an initial guess x_init

    :param x_init: initial x_0
    :param sino: Measurement sinogram
    :return:
    """
    img_shape = x_init.shape[1:]


    n_angles, det_count, forward_op = create_forward_op(img_shape)

    mu, bins = create_bins_and_mu()
    n_mats = mu.shape[1]
    # Initialize forward model
    forwardp = partial(forward,
                       bins=bins,
                       n_angles=n_angles,
                       det_count=det_count,
                       n_mats=n_mats,
                       image_shape=img_shape,
                       forward_op=forward_op,
                       mu=mu)

    # Initialize ProblemSetup
    P = ProblemSetup(
        y=sino,
        h=bins,
        proj=forward_op,
        img_shape=img_shape,
        sino_shape=sino.shape[1:],
        energies=[i for i in range(150)],
        n_mats=n_mats,
        mass_attn=mu,
        y_bar_func=forwardp
    )
    res = minimize(fun=partial(likelihood, P=P),
                   x0=x_init,
                   method='L-BFGS-B',
                   jac=True,
                   options={'disp':True}
                   )
    return res.x


def main():
    """If called as main: run LBFGS with sample image.

    :return:
    """
    img_shape = (128, 128)
    #img_shape = (3,3)  # For gradient verification

    # Create sample image
    x = np.zeros((2,*img_shape))
    x[0, 40:50, 40:50] = .1
    x[0, 60:80, 60:80] = .1
    x[1, 40:50, 60:80] = .1
    x[1, 60:80, 40:50] = .1

    mu, bins = create_bins_and_mu()

    n_angles, det_count, forward_op = create_forward_op(img_shape)

    # Initialize forward model
    forwardp = partial(forward,
                       bins=bins,
                       n_angles=n_angles,
                       det_count=det_count,
                       n_mats=2,
                       image_shape=img_shape,
                       forward_op=forward_op,
                       mu=mu)

    proj = forwardp(x)
    proj = np.random.poisson(proj)

    result = LBFGS(np.zeros(x.shape), proj)

    print(result)
    #grad_verification(partial(likelihood, P=P), (2,3,3))



if __name__ == '__main__':
    main()
