import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from Utils.my_utils_ddpm import x_pred, reverse_diffusion_from_x_pred, diffusion

def ODPS(Y, background, radon, binned_spectrum, Q, pixel_size, x_mass_density, rho, t_prime, step, x_scout, material_nn, mean_material, std_material, alpha, alpha_bar, sigma2, grad_approx, device):
    '''
    Y : [n_bin, n_angles, det_count]                Measures.
    background :                                    Dark current (set as zero for now).
    radon :                                         Torchradon operator.
    binned_spectrum :  [n_bin, 150]                 h_k functions; photon count spectrum as a function of the energy for the k bin. 
    Q : [150, n_mat]                                Material composition matrix.
    pixel_size : scalar                             Size (in cm) of pixels.
    x_mass density : [n_mat, img_size, img_size]    Ref material image.
    rho : [n_mat]                                   Material densities
    t_prime : int 0 and 999                         Start the reverse diffusion process.(999 -> full reverse diffusion process.)
    step : [1,n_mat,1,1]                            Data attachement gradient step size.
    x_scout : [n_mat, img_size, img_size]           Scout material image decomposition to diffuse to time t_prime before starting the reverse diffusion process.
    material_nn :                                   Neural Network used for generation of material images.
    mean and std material :                         Mean and Stardard deviation of material images.
    alpha, alpha_bar, sigma2                        Diffusion paramters (obtained with diffusion_parameters function).
    grad_approx : boolean                           If true, does not compute the full gradient of the data attachement term.
    device : gpu or cpu
    '''
    # Init
    n_bin = Y.shape[0]
    n_mat = x_mass_density.shape[1]
    img_size = x_mass_density.shape[-1]
    rho = rho[None,:,None,None].to(device).float()
    x_mass_density_np = x_mass_density.detach().cpu().numpy()

    x_odps = (x_scout/rho/pixel_size-mean_material)/std_material
    x_odps = diffusion(x_odps, alpha_bar[t_prime])

    x_temp = (x_odps * std_material + mean_material)*(pixel_size*rho)
    x_temp[x_temp<0] = 0
    x_temp = x_temp.detach().cpu().numpy()


    # Metrics
    PSNR_odps = torch.zeros([t_prime, n_mat], requires_grad=False)
    SSIM_odps = torch.zeros([t_prime, n_mat], requires_grad=False)

    for k in range(n_mat):
        PSNR_odps[0,k] = peak_signal_noise_ratio(x_mass_density_np[0,k],
                                                 x_temp[0,k],
                                                 data_range=x_mass_density_np[0,k].max())
        SSIM_odps[0,k] = structural_similarity(x_temp[0,k] ,
                                               x_mass_density_np[0,k],
                                               data_range=x_mass_density_np[0,k].max(),
                                               gradient=False)
    b, c, h, w = x_odps.shape
    for t in range(t_prime-1,0,-1):
        ###############
        # If gradient approximation is used (no automatic differentiation through neural network)
        if grad_approx:
            with torch.no_grad():
                x_odps = x_odps.float()

                # Prediction of clean image x_0 from noisy image x_t (using tweedie's formula and score approximation).
                x_0 = x_pred(x_odps, t, alpha_bar[t], img_size, material_nn, device).to(device)
                x_0_mat = (x_0* std_material + mean_material)*(pixel_size*rho)
                x_0_mat[x_0_mat<0] = 0
                
            with torch.enable_grad():
                x_0_mat = x_0_mat.requires_grad_()
                x_proj = radon.forward(x_0_mat).to(device)
                Qx_proj = Q @ x_proj.reshape(n_mat,x_proj.shape[-2]*x_proj.shape[-1])
                
                y_bar_x = binned_spectrum @ torch.exp(-Qx_proj)
                y_bar_x = y_bar_x.reshape(Y.shape)

                to_compute_grad = torch.nn.functional.mse_loss(y_bar_x, Y)

                # Computing gradient for each energy bins (using DPS approximation).
                # !!!!
                # Will be changed with exact gradient computation (?)
                # !!!!
                grad = torch.autograd.grad(outputs=to_compute_grad, inputs=x_0_mat, retain_graph=True, grad_outputs=torch.ones_like(to_compute_grad))
                grad = torch.stack(list(grad), dim=0)[0,:,:]

        ###############
       
        else:
        # If automatic differentiation through neural network:
            with torch.enable_grad():
                x_odps = x_odps.float()
                x_odps = x_odps.requires_grad_()

                # Prediction of clean image x_0 from noisy image x_t (using tweedie's formula and score approximation).
                x_0 = x_pred(x_odps, t, alpha_bar[t], img_size, material_nn, device).to(device)

                x_0_mat = (x_0* std_material + mean_material)*(pixel_size*rho)
                x_0_mat[x_0_mat<0] = 0

                x_proj = radon.forward(x_0_mat).to(device)
                Qx_proj = Q @ x_proj.reshape(n_mat,x_proj.shape[-2]*x_proj.shape[-1])
                
                y_bar_x = binned_spectrum @ torch.exp(-Qx_proj)
                y_bar_x = y_bar_x.reshape(Y.shape)

                to_compute_grad = torch.nn.functional.mse_loss(y_bar_x, Y)

                # Computing gradient for each energy bins (using DPS approximation).
                grad = torch.autograd.grad(outputs=to_compute_grad, inputs=x_odps, retain_graph=True, grad_outputs=torch.ones_like(to_compute_grad))
                grad = torch.stack(list(grad), dim=0)[0,:,:]
                
        ###############
        
        with torch.no_grad():
            # Gradient normalisation.
            norm = torch.zeros_like(x_odps, requires_grad=False).to(device)
            for k in range(n_mat):
                norm[0,k]=torch.linalg.norm(grad[0,k],ord='fro')


            # Unconditional reverse diffusion (only once per iteration).
            x_odps = reverse_diffusion_from_x_pred(x_odps, x_0, t, alpha[t], alpha_bar[t], sigma2[t], img_size, material_nn, device)


            # Gradient descent (conditional guidance).
            x_odps = x_odps - step*grad/norm


            # Un-standardisation
            x_odps_np= (x_odps* std_material + mean_material)*(pixel_size*rho)
            x_odps_np[x_odps_np<0] = 0                
            x_odps_np = x_odps_np.detach().cpu().numpy()

            # Computing metrics
            for m in range(n_mat):
                PSNR_odps[t_prime-t,m]  = peak_signal_noise_ratio(x_mass_density_np[0,m],x_odps_np[0,m],data_range=x_mass_density_np[0,m].max())
                SSIM_odps[t_prime-t,m] = structural_similarity(x_odps_np[0,m] ,x_mass_density_np[0,m], data_range=x_mass_density_np[0,m].max(), gradient=False)

    x_odps_final = torch.zeros_like(x_odps)
    x_odps_final = (x_odps* std_material + mean_material)*(pixel_size*rho)
    x_odps_final[x_odps_final<0] = 0

    return x_odps_final, PSNR_odps, SSIM_odps

    
    
