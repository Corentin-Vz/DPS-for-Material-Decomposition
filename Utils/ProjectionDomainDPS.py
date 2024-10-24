import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from Utils.my_utils_ddpm import x_pred, reverse_diffusion_from_x_pred, diffusion

def likelihood_2steps(Mat, Spect, Measures, a, device):
    """Calculate negative log likelihood and its gradient

    :param x: np.array: image of shape (#materials, *image_shape)
    :param P: ProblemSetup: see ProblemSetup dataclass
    :return: L(x), grad L(x)
    """

    a_reshaped = torch.tensor(a.reshape(Mat.n_mat, Measures.n_angles * Measures.det_count), device=device).float()

    
    a_comp = Mat.mass_attn @ a_reshaped  
    y_bar_a = Spect.binned_spectrum @ torch.exp(-a_comp)  
    y_bar_a = y_bar_a.reshape(Spect.n_bin, Measures.n_angles,Measures.det_count)
    
    L = torch.sum(-Measures.y * torch.log(y_bar_a) + y_bar_a).detach().cpu().numpy()

    grad_L = np.zeros(a_reshaped.shape)
    
    grad_phi = (1 - Measures.y / y_bar_a)

    att_term = torch.exp(-a_reshaped) # Shape = [batch, 150, n_angles, det_count]

    h_k_q = Mat.mass_attn[None,:,:] * Spect.binned_spectrum[:,:,None]   # Shape = [n_bin, 150, n_mat]
        
    inner = torch.sum(h_k_q.reshape(3,150,2,1) * att_term.reshape(1,1,Mat.n_mat,att_term.shape[-1]), dim=1)


    grad_L = torch.sum(-inner, dim=0).detach().cpu().numpy()  # sum over bins

    return L, grad_L.flatten()

def fullcost(x, Mat, Spect, Measures, beta, device):
    """Calculates full target function (likelihood + regularizers).

    :param x: input image
    :param P: Problem setup for likelihood calculation
    :param beta: huber prior weight
    :param delta: huber prior parameter
    :return:
    """
    fr, gr = imagePrior(x.reshape(Mat.n_mat, Measures.n_angles, Measures.det_count), 0.005)
    fl, gl = likelihood_2steps(Mat, Spect, Measures, x, device)
    ftot = fl + beta*fr
    gtot = gl + beta*gr
    return ftot, gtot

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
from scipy.optimize import minimize


def ProjectionDomainDPS_method(Mat, Spect, Measures, DPS, x_scout, beta, t_prime, step, device):
    '''
    Mat : Material class.
    Spect : Spectrum class.
    Measures : Measures class.
    DPS : DPS class    
    x_scout : tensor [1, n_mat, img_size, img_size]. Scout material images to diffuse to time t' before starting the reverse diffusion
    t_prime : scalar (between 1 and 999=t_max).
    step : tensor [n_mat]. Data attachement step for each material.
    grad_approx : boolean. If true, does not compute the full gradient of the data attachement term but only the log_likelihood gradient.
    device : gpu or cpu
    '''
    # STEP 1 : LBFGS + HUBER PRIOR FOR MATERIAL DECOMPOSITION IN PROJECTION DOMAIN.
    x_scout_proj = Measures.radon.forward(x_scout)
    res = minimize(fun=fullcost,               
               x0=x_scout_proj.detach().cpu().numpy().flatten(),
               args= (Mat, Spect, Measures, beta, device),
               method='L-BFGS-B',
               jac=True,
               options={'disp':False},
               bounds=((0,None),)
               )
    sino_step1 = torch.tensor(res.x.reshape(1,2,Measures.n_angles,Measures.det_count), device=device).float()
    
    # STEP 2 : DPS REGULARISATION FOR MATERIAL RECONSTRUCTIONS
    # Init
    x_mass_density_np = Mat.x_mass_densities.detach().cpu().numpy()

    x = (x_scout-DPS.mean)/DPS.std
    x = diffusion(x, DPS.alpha_bar[t_prime])

    x_temp = (x * DPS.std + DPS.mean)
    x_temp[x_temp<0] = 0
    x_temp = x_temp.detach().cpu().numpy()

    # Metrics
    PSNR = np.zeros([t_prime, Mat.n_mat])
    SSIM = np.zeros([t_prime, Mat.n_mat])

    for k in range(Mat.n_mat):
        PSNR[0,k] = peak_signal_noise_ratio(x_mass_density_np[0,k],
                                                 x_temp[0,k],
                                                 data_range=x_mass_density_np[0,k].max())
        SSIM[0,k] = structural_similarity(x_temp[0,k] ,
                                               x_mass_density_np[0,k],
                                               data_range=x_mass_density_np[0,k].max(),
                                               gradient=False)

    for t in range(t_prime-1,0,-1):

        with torch.enable_grad():
            x = x.float()
            x = x.requires_grad_()

            # Prediction of clean image x_0 from noisy image x_t (using tweedie's formula and score approximation).
            x_0 = x_pred(x, t, DPS.alpha_bar[t], Mat.img_size, DPS.nn, device).to(device)

            x_0_mat = (x_0* DPS.std + DPS.mean)
            x_0_mat[x_0_mat<0] = 0

            x_proj = Measures.radon.forward(x_0_mat).to(device)

            to_compute_grad = torch.nn.functional.mse_loss(x_proj, sino_step1)

            # Computing gradient for each energy bins (using DPS approximation).
            grad = torch.autograd.grad(outputs=to_compute_grad, inputs=x, retain_graph=True, grad_outputs=torch.ones_like(to_compute_grad))
            grad = torch.stack(list(grad), dim=0)[0,:,:]

        
        with torch.no_grad():
            # Gradient normalisation.
            norm = torch.zeros_like(x, requires_grad=False).to(device)
            for k in range(Mat.n_mat):
                norm[0,k]=torch.linalg.norm(grad[0,k],ord='fro')


            # Unconditional reverse diffusion (only once per iteration).
            x = reverse_diffusion_from_x_pred(x, x_0, t, DPS.alpha[t], DPS.alpha_bar[t], DPS.sigma2[t], Mat.img_size, DPS.nn, device)


            # Gradient descent (conditional guidance).
            x = x - step*grad/norm


            # Un-standardisation
            x_np= (x* DPS.std + DPS.mean)
            x_np[x_np<0] = 0                
            x_np = x_np.detach().cpu().numpy()

            # Computing metrics
            for m in range(Mat.n_mat):
                PSNR[t_prime-t,m]  = peak_signal_noise_ratio(x_mass_density_np[0,m],x_np[0,m],data_range=x_mass_density_np[0,m].max())
                SSIM[t_prime-t,m] = structural_similarity(x_np[0,m] ,x_mass_density_np[0,m], data_range=x_mass_density_np[0,m].max(), gradient=False)

    x_final = torch.zeros_like(x)
    x_final = (x* DPS.std + DPS.mean)
    x_final[x_final<0] = 0

    return x_final, PSNR, SSIM
