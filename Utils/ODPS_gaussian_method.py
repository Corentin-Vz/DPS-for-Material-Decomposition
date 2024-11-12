from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from Utils.my_utils_ddpm import x_pred, reverse_diffusion_from_x_pred, diffusion
from Utils.forward_model import forward_mat_op
import numpy as np
import torch

def ODPS_gaussian_method(Mat, Spect, Measures, ODPS, x_scout, t_prime, step, device):
    '''
    Mat : Material class.
    Spect : Spectrum class.
    Measures : Measures class.
    ODPS : ODPS class    
    x_scout : tensor [1, n_mat, img_size, img_size]. Scout material images to diffuse to time t' before starting the reverse diffusion
    t_prime : scalar (between 1 and 999=t_max).
    step : tensor [n_mat]. Data attachement step for each material.
    grad_approx : boolean. If true, does not compute the full gradient of the data attachement term but only the log_likelihood gradient.
    device : gpu or cpu
    '''
    # Init
    x_mass_density_np = Mat.x_mass_densities.detach().cpu().numpy()

    x_odps = (x_scout-ODPS.mean)/ODPS.std
    x_odps = diffusion(x_odps, ODPS.alpha_bar[t_prime])

    x_temp = (x_odps * ODPS.std + ODPS.mean)
    x_temp[x_temp<0] = 0
    x_temp = x_temp.detach().cpu().numpy()
    
    step = step.reshape(1,Mat.n_mat,1,1)

    # Metrics
    PSNR_odps = np.zeros([t_prime, Mat.n_mat])
    SSIM_odps = np.zeros([t_prime, Mat.n_mat])

    for k in range(Mat.n_mat):
        PSNR_odps[0,k] = peak_signal_noise_ratio(x_mass_density_np[0,k],
                                                 x_temp[0,k],
                                                 data_range=x_mass_density_np[0,k].max())
        SSIM_odps[0,k] = structural_similarity(x_temp[0,k] ,
                                               x_mass_density_np[0,k],
                                               data_range=x_mass_density_np[0,k].max(),
                                               gradient=False)

    for t in range(t_prime-1,0,-1):
        ###############
        # If gradient approximation is used (no automatic differentiation through neural network)
       
        with torch.no_grad():
            x_odps = x_odps.float()

            # Prediction of clean image x_0 from noisy image x_t (using tweedie's formula and score approximation).
            x_0 = x_pred(x_odps, t, ODPS.alpha_bar[t], Mat.img_size, ODPS.nn, device).to(device)
            x_0_mat = (x_0* ODPS.std + ODPS.mean)
            x_0_mat[x_0_mat<0] = 0

            L, grad = likelihood_gaussian(x_0_mat, Mat, Spect, Measures, device)
        ###############        
        with torch.no_grad():
            # Gradient normalisation.
            norm = torch.zeros_like(x_odps, requires_grad=False).to(device)
            for k in range(Mat.n_mat):
                norm[0,k]=torch.linalg.norm(grad[0,k],ord='fro')


            # Unconditional reverse diffusion (only once per iteration).
            x_odps = reverse_diffusion_from_x_pred(x_odps, x_0, t, ODPS.alpha[t], ODPS.alpha_bar[t], ODPS.sigma2[t], Mat.img_size, ODPS.nn, device)


            # Gradient descent (conditional guidance).
            x_odps = x_odps - step*grad/norm


            # Un-standardisation
            x_odps_np= (x_odps* ODPS.std + ODPS.mean)
            x_odps_np[x_odps_np<0] = 0                
            x_odps_np = x_odps_np.detach().cpu().numpy()

            # Computing metrics
            for m in range(Mat.n_mat):
                PSNR_odps[t_prime-t,m]  = peak_signal_noise_ratio(x_mass_density_np[0,m],x_odps_np[0,m],data_range=x_mass_density_np[0,m].max())
                SSIM_odps[t_prime-t,m] = structural_similarity(x_odps_np[0,m] ,x_mass_density_np[0,m], data_range=x_mass_density_np[0,m].max(), gradient=False)

    x_odps_final = torch.zeros_like(x_odps)
    x_odps_final = (x_odps* ODPS.std + ODPS.mean)
    x_odps_final[x_odps_final<0] = 0

    return x_odps_final, PSNR_odps, SSIM_odps

def likelihood_gaussian(x, Mat, Spect, Measures, device):
    """Calculate negative log likelihood and its gradient

    :param x: np.array: image of shape (#materials, *image_shape)
    :param P: ProblemSetup: see ProblemSetup dataclass
    :return: L(x), grad L(x)
    """
    
    y_bar_x = forward_mat_op(x, Mat.mass_attn, Spect.binned_spectrum, Measures.background, Measures.radon, False, device)
    diff = (y_bar_x-Measures.y)
    L = 0.5* torch.sum((diff/Measures.y)**2)

    grad_L = torch.zeros_like(x)
    
    weigths = (1 / Measures.y)**2

    combined_mats = Mat.mass_attn @ x.reshape(Mat.n_mat, Mat.img_size**2)  # Combine materials
    exponent = Measures.radon.forward(combined_mats.reshape(1,150, Mat.img_size, Mat.img_size))
    att_term = torch.exp(-exponent) # Shape = [batch, 150, n_angles, det_count]

    grad_L = torch.zeros(1,Mat.n_mat,Mat.img_size,Mat.img_size, device=device)
    for idm in range(Mat.n_mat):
        h_k_q = Mat.mass_attn[:,idm][None,:] * Spect.binned_spectrum   # Shape = [n_bin, 150]
        
        inner = torch.sum(h_k_q.reshape(Spect.n_bin, 150, 1, 1) * att_term * diff.reshape(diff.shape[0], 1, diff.shape[1], diff.shape[2]), dim=1)  # sum over energies; this is the integral over E

        backprojected= Measures.radon.backward(inner*weigths)  # backproject integral * phi'

        grad_L[0,idm] = torch.sum(-backprojected, dim=0)  # sum over bins

    return  L, grad_L
