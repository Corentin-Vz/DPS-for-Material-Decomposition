from scipy.optimize import minimize
import itertools
import torch
import numpy as np
from Utils.forward_model import forward_mat_op
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def likelihood(Mat, Spect, Measures, x, device):
    """Calculate negative log likelihood and its gradient

    :param x: np.array: image of shape (#materials, *image_shape)
    :param P: ProblemSetup: see ProblemSetup dataclass
    :return: L(x), grad L(x)
    """

    x_shaped = torch.tensor(x.reshape(1, Mat.n_mat, Mat.img_size, Mat.img_size), device=device).float()
    
    y_bar_x = forward_mat_op(x_shaped, Mat.mass_attn, Spect.binned_spectrum, Measures.background, Measures.radon, False, device)

    L = torch.sum(-Measures.y * torch.log(y_bar_x) + y_bar_x).detach().cpu().numpy()

    grad_L = np.zeros(x_shaped.shape)
    
    grad_phi = (1 - Measures.y / y_bar_x)

    combined_mats = Mat.mass_attn @ x_shaped.reshape(Mat.n_mat, Mat.img_size**2)  # Combine materials
    exponent = Measures.radon.forward(combined_mats.reshape(1,150, Mat.img_size, Mat.img_size))
    att_term = torch.exp(-exponent) # Shape = [batch, 150, n_angles, det_count]

    for idm in range(Mat.n_mat):
        h_k_q = Mat.mass_attn[:,idm][None,:] * Spect.binned_spectrum   # Shape = [n_bin, 150]
        
        inner = torch.sum(h_k_q.reshape(Spect.n_bin, 150, 1, 1) * att_term, dim=1)  # sum over energies; this is the integral over E

        backprojected= Measures.radon.backward(inner*grad_phi)  # backproject integral * phi'

        grad_L[0,idm] = torch.sum(-backprojected, dim=0).detach().cpu().numpy()  # sum over bins

    return L, grad_L.flatten()

def fullcost(x, Mat, Spect, Measures, beta, gamma, device):
    """Calculates full target function (likelihood + regularizers).

    :param x: input image
    :param P: Problem setup for likelihood calculation
    :param beta: huber prior weight
    :param gamma: inner product weight
    :param delta: huber prior parameter
    :return:
    """
    fr, gr = imagePrior(x.reshape(Mat.n_mat, Mat.img_size, Mat.img_size), 0.005)
    fi, gi = inner_prod_reg(x.reshape(Mat.n_mat, Mat.img_size, Mat.img_size), np.ones(len(x)))
    fl, gl = likelihood(Mat, Spect, Measures, x, device)
    ftot = fl + beta*fr + gamma*fi
    gtot = gl + beta*gr + gamma*gi
    return ftot, gtot

def inner_prod_reg(im, weights):
    """Inner product regularizer. Supports arbitrary amount of materials, returns sum of pair-wise inner product.

    :param im: Input image. Inner product will be calculated in dimension 0.
    :param weights: np.array, shape (len(im)). Allows weighting of different channels. Equal weighting by default.
    :return: f(im), grad f(im)
    """
#     if not weights:
#         weights = np.ones(len(im))
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

def OneStepLBFGS(Mat, Spect, Measures, x_scout, beta, gamma, device):
    
    res = minimize(fun=fullcost,               
               x0=x_scout.detach().cpu().numpy().flatten(),
               args= (Mat, Spect, Measures, beta, gamma, device),
               method='L-BFGS-B',
               jac=True,
               options={'disp':False},
               bounds=((0,None),)
               )
    
    x_lbfgs = res.x.reshape(1, Mat.n_mat, Mat.img_size, Mat.img_size)
    

    # Computing metrics
    x_mass_densities_np = Mat.x_mass_densities.detach().cpu().numpy()
    PSNR_OnestepLBFGS = np.zeros(Mat.n_mat)
    SSIM_OnestepLBFGS = np.zeros(Mat.n_mat)
    for m in range(Mat.n_mat):
        PSNR_OnestepLBFGS[m]  = peak_signal_noise_ratio(x_mass_densities_np[0,m],x_lbfgs[0,m],data_range=x_mass_densities_np[0,m].max())
        SSIM_OnestepLBFGS[m] = structural_similarity(x_lbfgs[0,m] ,x_mass_densities_np[0,m], data_range=x_mass_densities_np[0,m].max(), gradient=False)
        
    return x_lbfgs, PSNR_OnestepLBFGS, SSIM_OnestepLBFGS
