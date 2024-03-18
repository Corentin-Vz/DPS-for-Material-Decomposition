import torch
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import copy
import time

class ParamRecon:
    def __init__(self, I, nb_iter, bckg, prior, beta, delta):
        self.I = I
        self.nb_iter = nb_iter
        self.bckg = bckg
        self.prior = prior
        self.beta = beta
        self.delta = delta

    def get_paramRecon(self):
        return self

class ParamRecon_dTV(ParamRecon):
    def __init__(self, I, nb_iter, bckg, prior, beta, delta, nInnerIter, theta, eta, eps):
        ParamRecon.__init__(self, I, nb_iter, bckg, prior, beta, delta)
        self.nInnerIter = nInnerIter
        self.theta = theta
        self.eta = eta
        self.eps = eps


def compute_grad(x, device):
    # Input shape : [1,1,img_size,img_size]
    # Output shape : [1,1,img_size,img_size, 2]
    Dx = torch.zeros(x.shape[0],x.shape[1],x.shape[2],x.shape[3],2, device=device)
    Dx[:,:,:, :, 0] = x - torch.roll(torch.roll(x, 1, dims=2), 0, dims=1)
    Dx[:,:,:, :, 1] = x - torch.roll(torch.roll(x, 1, dims=3), 0, dims=1)
    return Dx

def compute_grad_back(y, device):
    # Input shape : [1,1,img_size,img_size, 2]
    # Output shape : [1,1,img_size,img_size]
    x = torch.zeros([y.shape[0],y.shape[1],y.shape[2], y.shape[3]], device=device)
    x = x + y[:, :, :, :, 0] - torch.roll(torch.roll(y[:, :, :, :, 0], -1, dims=2), 0, dims=3)
    x = x + y[:, :, :, :, 1] - torch.roll(torch.roll(y[:, :, :, :, 1], 0, dims=2), -1, dims=3)
    return x

def computeP(u, xi, device):
    res = torch.zeros(u.shape, device=device)
    res[:,:,:,:,0] = u[:,:,:,:,0] - u[:,:,:,:,0] * (xi[:,:,:,:,0]**2) - xi[:,:,:,:,0]*xi[:,:,:,:,1]*u[:,:,:,:,1]
    res[:,:,:,:,1] = u[:,:,:,:,1] - u[:,:,:,:,1] * (xi[:,:,:,:,1]**2) - xi[:,:,:,:,0]*xi[:,:,:,:,1]*u[:,:,:,:,0]
    return res


def recoCT_dTV(y, x_prior, img_size, radon, paramRecon_dTV, device):
    assert paramRecon_dTV.prior == 'dTV', "paramRecon.prior should be dTV"

    diff = y - paramRecon_dTV.bckg
    b = torch.where(diff > 0, torch.log(paramRecon_dTV.I / diff), 0).to(device)
    w = torch.where(diff > 0, torch.square(diff) / y, 0).to(device)

    N = img_size
    
    beta = paramRecon_dTV.beta
    theta = paramRecon_dTV.theta

    w = [i / paramRecon_dTV.I for i in w]
    w = torch.stack(w).to(device)

    
    beta = beta / paramRecon_dTV.I


    # compute directional information xi from x_prior
    # xi = eta times the normalized gradient of the prior
    grad_x_prior = compute_grad(x_prior.to(device), device).to(device)
    norm_grad_x_prior_eps = torch.sum(torch.sqrt(grad_x_prior[:,:,:,:,0]**2 + grad_x_prior[:,:,:,:,1]**2 + paramRecon_dTV.eps**2)).to(device)
    eta = paramRecon_dTV.eta
    xi = torch.zeros(grad_x_prior.shape, device=device)
    xi = eta * grad_x_prior / norm_grad_x_prior_eps
    
    
    
    x = torch.randn(1,1,N, N, device=device)

    for k in range(100):
        x = compute_grad_back(computeP(computeP(compute_grad(x,device),xi,device),xi,device),device)
        normalisation = torch.sqrt(torch.sum(x ** 2))
        x = x / normalisation
        s = torch.sqrt(torch.sum(computeP(compute_grad(x,device),xi,device)**2))

    L = s
    sigma = 1 / L
    tau = 1 / L

    x = torch.zeros(1,1,N,N, device=device)

    xbar = copy.deepcopy(x)

    z = torch.zeros(computeP(compute_grad(xbar,device),xi,device).shape, device=device)
    
    proj_ones = radon.forward(torch.ones(1,1,N, N, device=device))
    D_rec = radon.backward((w * proj_ones).float())
        
    for i in range(paramRecon_dTV.nb_iter):
        z_temp = z + sigma * computeP(compute_grad(xbar, device),xi,device)
        normZ = torch.sqrt(z_temp[:,:,:,:,0]**2 + z_temp[:,:,:,:,1]**2)
        z = torch.zeros(z_temp.shape, device=device)
        z[:, :,:,:, 0] = torch.where(normZ > beta, beta, normZ) * (z_temp[:, :,:,:, 0] / normZ)
        z[:, :,:,:, 1] = torch.where(normZ > beta, beta, normZ) * (z_temp[:, :,:,:, 1] / normZ)
        z = torch.where(torch.isnan(z), 0, z)

        x_old = copy.deepcopy(x)
        x_temp = x_old - tau * compute_grad_back(computeP(z,xi,device),device)

        for inner_k in range(paramRecon_dTV.nInnerIter):
            # FBS algo
            # Forward:
            x_rec = x - radon.backward(w * (radon.forward(x) - b).float()) / D_rec
            # Backward
            x = (D_rec * x_rec + x_temp / tau) / (D_rec + 1 / tau)
            x = torch.where(x > 0, x, 0)

        xbar = x + theta * (x - x_old)
        
    return xbar

