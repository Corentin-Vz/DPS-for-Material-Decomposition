import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from Utils.my_utils_ddpm import x_pred, reverse_diffusion_from_x_pred, diffusion

def TDPS_method(Mat, Spect, Measures, TDPS, x_scout, t_prime, step, device):
    '''
    Mat : Material class.
    Spect : Spectrum class.
    Measures : Measures class.
    TDPS : ODPS class    
    x_scout : tensor [1, n_mat, img_size, img_size]. Scout material images to diffuse to time t' before starting the reverse diffusion
    t_prime : scalar (between 1 and 999=t_max).
    step : tensor [n_mat]. Data attachement step for each material.
    device : gpu or cpu
    '''
    
    sino_approx_spectral = torch.log(torch.sum(Spect.binned_spectrum, dim=1)[:,None,None]/Measures.y)/Mat.pixel_size
    sino_approx_spectral[Measures.y==0] = 0
    sino_approx_spectral = sino_approx_spectral[None]

    x_mass_density_np = Mat.x_mass_densities.detach().cpu().numpy()

    x_tdps = Mat.mass_attn_pseudo_spectral @  (x_scout.reshape(2, Mat.img_size**2)/Mat.pixel_size ).float()

    x_tdps = (x_tdps.reshape(1, Spect.n_bin, Mat.img_size, Mat.img_size)-TDPS.mean)/TDPS.std


    x_tdps = diffusion(x_tdps, TDPS.alpha_bar[t_prime])
    x_temp = TDPS.std * x_tdps + TDPS.mean
    x_temp[x_temp<0]=0
    x_temp =  torch.linalg.lstsq(Mat.mass_attn_pseudo_spectral, 
                Mat.pixel_size * x_tdps[0,:].reshape(Spect.n_bin, Mat.img_size**2),
                driver="gels").solution.reshape(2, Mat.img_size, Mat.img_size)[None,:,:,:]
    PSNR_tdps = np.zeros([t_prime, Mat.n_mat])
    SSIM_tdps = np.zeros([t_prime, Mat.n_mat])

    for k in range(Mat.n_mat):
        PSNR_tdps[0,k] = peak_signal_noise_ratio(x_mass_density_np[0,k], x_temp[0,k].detach().cpu().numpy(),
                                                  data_range=x_mass_density_np[0,k].max())
        SSIM_tdps[0,k] = structural_similarity(x_temp[0,k].detach().cpu().numpy(), x_mass_density_np[0,k],
                                                data_range=x_mass_density_np[0,k].max(), gradient=False)


    for t in range(t_prime,0,-1):
            x_tdps = x_tdps.requires_grad_().float()

            # Prediction of clean image x_0 from noisy image x_t (using tweedie's formula and score approximation).
            x_0 = x_pred(x_tdps, t, TDPS.alpha_bar[t], Mat.img_size, TDPS.nn, device).to(device)
            x_0_LAC = TDPS.std*x_0 + TDPS.mean
            x_0_LAC[x_0_LAC<0] =0

            # Projection of x_0_LAC according to the forward model.
            radon_x_pred = Measures.radon.forward(x_0_LAC).to(device)             

            to_compute_grad = torch.nn.functional.mse_loss(radon_x_pred, sino_approx_spectral)

            # Computing gradient for each energy bins (using DPS approximation).
            grad = torch.autograd.grad(outputs=to_compute_grad, inputs=x_tdps,
                                       retain_graph=False, grad_outputs=torch.ones_like(to_compute_grad))
            grad = torch.stack(list(grad), dim=0)[0,:,:]

            with torch.no_grad():
                # Gradient normalisation.
                norm = torch.zeros_like(x_tdps, requires_grad=False).to(device)
                for k in range(Spect.n_bin):
                    norm[0,k]=torch.linalg.norm(grad[0,k],ord='fro')



                x_tdps = reverse_diffusion_from_x_pred(x_tdps.detach(), x_0, t, TDPS.alpha[t],
                                                        TDPS.alpha_bar[t], TDPS.sigma2[t], Mat.img_size, TDPS.nn, device)

                x_tdps = x_tdps - step*grad/norm


                x_temp_spectral = (TDPS.std*x_tdps.detach() + TDPS.mean)
                x_temp_spectral[x_temp_spectral<0] = 0
                x_temp_spectral = x_temp_spectral.float()

                x_temp =  torch.linalg.lstsq(Mat.mass_attn_pseudo_spectral, 
                            Mat.pixel_size * x_temp_spectral[0,:].reshape(Spect.n_bin, Mat.img_size**2),
                            driver="gels").solution.reshape(2, Mat.img_size, Mat.img_size)[None,:,:,:]
                x_temp =  x_temp.detach().cpu().numpy()
                x_temp[x_temp<0]=0
                x_temp_spectral = x_temp_spectral.detach().cpu().numpy()
                x_temp_spectral[x_temp_spectral<0]=0

                # Metrics
                for k in range(Mat.n_mat):
                    PSNR_tdps[t_prime-t,k] = peak_signal_noise_ratio(x_mass_density_np[0,k],x_temp[0,k],
                                                                        data_range=x_mass_density_np[0,k].max())
                    SSIM_tdps[t_prime-t,k] = structural_similarity(x_temp[0,k], x_mass_density_np[0,k],
                                                                      data_range=x_mass_density_np[0,k].max(), gradient=False)

    x_tdps_final = torch.zeros_like(x_tdps)
    x_tdps_final = (TDPS.std*x_tdps.detach() + TDPS.mean)
    x_tdps_final[x_tdps_final<0] = 0

    x_tdps_mat = torch.linalg.lstsq(Mat.mass_attn_pseudo_spectral, 
                (Mat.pixel_size * x_tdps_final[0,:].reshape(Spect.n_bin, Mat.img_size**2)).float(),
                driver="gels").solution.reshape(2, Mat.img_size, Mat.img_size)[None,:,:,:]
    x_tdps_mat = x_tdps_mat.detach().cpu().numpy()
    x_tdps_mat[x_tdps_mat<0]=0

    return x_tdps_mat, x_tdps_final, PSNR_tdps, SSIM_tdps
    
