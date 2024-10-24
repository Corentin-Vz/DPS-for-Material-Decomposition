import torch
import numpy as np
from Utils.my_utils_radon import get_filter, filter_sinogram
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def FBP_Pinv(Mat, Spect, Measures):
    
    sino_approx_spectral = torch.log(torch.sum(Spect.binned_spectrum, dim=1)[:,None,None]/Measures.y)
    sino_approx_spectral[Measures.y==0] = 0
    sino_approx_spectral = sino_approx_spectral[None]
    fbp = Measures.radon.backward(filter_sinogram(sino_approx_spectral))


    x_mat = torch.linalg.lstsq(Mat.mass_attn_pseudo_spectral, 
                               fbp[0,:].reshape(Spect.n_bin, Mat.img_size**2),
                               driver="gels").solution.reshape(Mat.n_mat, Mat.img_size, Mat.img_size)[None,:,:,:]

    # Computing metrics
    PSNR_fbp_pinv = np.zeros(Mat.n_mat)
    SSIM_fbp_pinv = np.zeros(Mat.n_mat)
    for k in range(Mat.n_mat):
        PSNR_fbp_pinv[k] = peak_signal_noise_ratio(Mat.x_mass_densities.detach().cpu().numpy()[0,k],x_mat[0,k].detach().cpu().numpy(),data_range=Mat.x_mass_densities[0,k].detach().cpu().numpy().max())
        SSIM_fbp_pinv[k] = structural_similarity(x_mat[0,k].detach().cpu().numpy(), Mat.x_mass_densities[0,k].detach().cpu().numpy(), data_range=Mat.x_mass_densities[0,k].detach().cpu().numpy().max(), gradient=False)
        
    return x_mat, PSNR_fbp_pinv, SSIM_fbp_pinv
        

