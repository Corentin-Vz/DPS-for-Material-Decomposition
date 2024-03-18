import os 
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from Data.DataLoader_material import DataLoader_material
from Data.DataLoader_spectral import DataLoader_spectral
from my_utils_ddpm import diffusion_parameters, diffusion, reverse_diffusion, x_pred
from my_utils_radon import get_filter, filter_sinogram
from torch_radon import Radon, RadonFanbeam
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import utils_dtv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

'''
HYPERPARAMETERS FOR THIS MULTI TEST SCRIPT :

n_iter : Number of iteration for one reverse guided diffusion, i.e the diffusion step where the algorithm starts. It is an integer between 0 and 999 (we used a 1000 step diffusion process during training). The algorithm goes from n_iter to 0. ( i.e. performs reverse diffusion)

step_list_*DPS : gradient step during the conditionnal guidance for method *DPS. 

I_list : List of source photon counts for each energy bins. (Right now, each energy bins have the same source intensity.)

n_recon : When multiplied by batch_size, gives the total number of slices reconstructed for each hyperparameter possibilities.
    '''

n_recon = 15
n_iter_list = [999]
step_list_tdps = torch.tensor([0.3,0.3,0.3],device=device)
step_list_odps =  torch.tensor([0.45,0.45], device=device)
number_grad_descent = 1
I_list = [5000, 10000]
batch_size = 1 # Only batch size of 1 is currently working. (TODO)
imshow = False

# Transformation to apply to loaded images      
my_transforms = transforms.Compose(
    [transforms.ToTensor(),
     torchvision.transforms.RandomVerticalFlip(p=1) # (In order to have the bed on the bottom of the image)
    ]
)     

# Data loader (we use the test dataset here)
material_list = ['Bones','Soft Tissues']
n_mat = len(material_list)
patient_list = [12]
pixel_size = 1
data_material = DataLoader_material(img_dir="Data/", material_list=material_list, patient_list=patient_list, transform=my_transforms)
Data_material_test = DataLoader(data_material, batch_size=batch_size, shuffle=True)
mean_material = torch.tensor(np.load('Data/mean_material.npy'), device=device)[None,:,None,None]
std_material = torch.tensor(np.load('Data/std_material.npy'), device=device)[None,:,None,None]

energy_list=[40,80,120]
n_bin = len(energy_list)
data_spectral = DataLoader_spectral(img_dir="Data/", energy_list=energy_list, patient_list=patient_list, transform=my_transforms)
Data_spectral_test = DataLoader(data_spectral, batch_size=batch_size, shuffle=True)
mean_spectral = torch.tensor(np.load('Data/mean_spectral.npy'), device=device)[None,:,None,None]
std_spectral = torch.tensor(np.load('Data/std_spectral.npy'), device=device)[None,:,None,None]


# Computing material composition matrix (40, 60, 80, 100, 120, 140 kev):
M = torch.zeros(6,n_mat)
M[:,0] = torch.tensor([6.655E-01, 3.148E-01, 2.229E-01, 1.855E-01, 0, 1.480E-01]) *1.920E+00 # Bone, cortical
M[:,1] = torch.tensor([2.688E-01, 2.048E-01, 1.823E-01, 1.693E-01, 0, 1.492E-01]) *1.060E+00 # Soft tissues
M[4] = 3/5*M[3] + 2/5*M[5] 

 # Selecting row corresponding to energy_list: ( in this case 40, 80 and 120 keV)
M = M[(0,2,4),:]
M = (M*pixel_size).float().to(device)

# Diffusion model parameters
T = 1000
alpha, alpha_bar = diffusion_parameters(T)
sigma2 = (1-alpha)*(1-alpha_bar/alpha)/(1-alpha_bar)
img_size = 512

# # Neural networks:
# UNet Spectral
from neural_networks.UNet import UNet
spectral_nn = UNet(image_channels = n_bin, n_channels=8)
spectral_nn = spectral_nn.to(device)
ckpt_name = "checkpoints_spectral/nn_weights/spectral.pth"
spectral_nn.load_state_dict(torch.load(ckpt_name))

# UNet Material
from neural_networks.UNet import UNet
material_nn = UNet(image_channels = n_mat, n_channels=8)
material_nn = material_nn.to(device)
ckpt_name = "checkpoints_material/nn_weights/material.pth"
material_nn.load_state_dict(torch.load(ckpt_name))


# TORCH RADON
n_angles = 120 # define measuring angles
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
pix_size =  1
det_width = 1.2/pix_size
det_count = 750 
source_origin =  600/ pix_size
origin_det = 600 / pix_size 
radon = RadonFanbeam(resolution=img_size, angles=angles,det_count= det_count, det_spacing=det_width, source_distance=source_origin, det_distance=origin_det, clip_to_circle=False)
background = 0 #TODO


for n in range(n_recon): 
    print(n)
#     # Selecting a new slice
    x_true_spectral, index = next(iter(Data_spectral_test))
    x_true_spectral = (x_true_spectral*pixel_size).to(device).float()

    x_true_material, index = data_material[index.item()]
    x_true_material = x_true_material[None,:,:,:].to(device).float()
    x_true_material_np = x_true_material.detach().cpu().numpy()

   # Saving .npy file
    file_path = f'res/{n}'
    if not os.path.exists(file_path): 
        os.makedirs(file_path) 
    np.save(f'res/{n}/x_true_spectral.npy',x_true_spectral.detach().cpu().numpy())
    np.save(f'res/{n}/x_true_material.npy',x_true_material_np)
    np.save(f'res/{n}/index.npy', index)
    x_true_material_np = np.load(f'res/{n}/x_true_material.npy')
    
    for I0 in I_list:

        I = [I0, I0, I0]
        I = torch.tensor(I, device=device)[None,:,None,None] # (Broadcast)
        
        measures = (torch.poisson(I*torch.exp(-radon.forward(x_true_spectral))+background)).float()        

#         ###########################
#         # DTV + pseudo inv method #
#         ###########################
        # Reconstruction of an image prior by WLS for DTV
        x_prior = []
        dummy_image = torch.zeros(1,1,img_size,img_size)
        measures_all = torch.sum(measures, dim=1, keepdim=True)
        sino_approx_all = torch.log(torch.sum(I)/measures_all)
        sino_approx_all[torch.isinf(sino_approx_all)]=0

        paramRecon = utils_dtv.ParamRecon_dTV(torch.sum(I).item(), 100, bckg=background, prior='dTV', beta=0, delta=0.001, nInnerIter=1, theta=1, eta=0., eps=0.001).get_paramRecon()
        x_prior = utils_dtv.recoCT_dTV(measures_all,dummy_image, img_size, radon, paramRecon, device)

        x_prior_np = x_prior.detach().cpu().numpy()
        np.save(f'res/{n}/{I0}/DTV_prior.npy', x_prior_np)
        
        # reconstruction by DTV
        beta = [0.2*I0,0.21*I0,0.22*I0] # weight for the regularization
        x_DTV = []

        for i_e in range(n_bin):

            paramRecon = utils_dtv.ParamRecon_dTV(I[0,i_e,0,0].item(), 6000, bckg=background, prior='dTV', beta=beta[i_e], delta=0.001, nInnerIter=1, theta=1, eta=1, eps=0.001).get_paramRecon()
            x = utils_dtv.recoCT_dTV(measures[0,i_e,:,:],x_prior, img_size, radon, paramRecon, device)

            x_DTV.append(x)



        x_DTV_torch = torch.stack(x_DTV).squeeze()[None,:,:,:]
        x_DTV_np = x_DTV_torch.detach().cpu().numpy()

        # Computes pseudo inverse of M and applies it to X_lac in order to obtain X_mat
        x_DTV_md = torch.linalg.lstsq(M, x_DTV_torch[0].reshape(n_bin, img_size**2), driver="gels").solution
        x_DTV_md[x_DTV_md<0] = 0
        x_DTV_md = x_DTV_md.reshape(n_mat, img_size, img_size)
        x_DTV_md = x_DTV_md[None,:,:,:]
        x_DTV_md_np= x_DTV_md.detach().cpu().numpy()


        PSNR_DTV_Pinv = np.zeros(n_mat)
        SSIM_DTV_Pinv = np.zeros(n_mat)
        for b in range(n_mat):
            PSNR_DTV_Pinv[b]  = peak_signal_noise_ratio(x_true_material_np[0,b],x_DTV_md_np[0,b],data_range=x_true_material_np[0,b].max())
            SSIM_DTV_Pinv[b] = structural_similarity(x_DTV_md_np[0,b] ,x_true_material_np[0,b], data_range=x_true_material_np[0,b].max(), gradient=False)

        np.save(f'res/{n}/{I0}/DTV.npy', x_DTV_np)
        np.save(f'res/{n}/{I0}/DTV_Pinv.npy', x_DTV_md_np)
        np.save(f'res/{n}/{I0}/PSNR_DTV_Pinv.npy', PSNR_DTV_Pinv)
        np.save(f'res/{n}/{I0}/SSIM_DTV_Pinv.npy', SSIM_DTV_Pinv)    

#         recon = x_DTV_torch
#         recon_md = x_DTV_md

        for n_iter in n_iter_list:
            print(f'Starting : {n} slice, {I0} photons, {number_grad_descent} grad descent, {n_iter} iteration')
# # #             ##############################################################
# # #             # Two-step material decomp with diffusion posterior sampling #
# # #             ##############################################################

            # STEP 1 : sampling from spectral image given measurements (DPS)
            
            # Standardisation of scout recon and diffusion to time t=n_iter.
            x_tdps = (x_DTV_torch-mean_spectral)/std_spectral
            x_tdps = diffusion(x_tdps, alpha_bar[n_iter])

            for t in range(n_iter-1,0,-1):
                for step_k in range(number_grad_descent):
                    x_tdps = x_tdps.requires_grad_()

                    # Prediction of clean image x_0 from noisy image x_t (using tweedie's formula and score approximation).
                    x_0 = (x_pred(x_tdps, t, alpha_bar[t], img_size, spectral_nn, device)).to(device)

                    # From standardized to LAC coefficients using mean and std computed on the training data.
                    x_0_LAC = x_0*std_spectral + mean_spectral
                    x_0_LAC[x_0_LAC<0] = 0

                    # Projection of x_0_LAC according to the forward model.
                    radon_x_pred = radon.forward(x_0_LAC).to(device)
                    A_x0 = I0*torch.exp(-radon_x_pred)                  

                    #  Data attachement term.
                    loss_matrix = torch.matmul(torch.transpose((measures-A_x0), 2, 3),(measures-A_x0)/(2*measures))
                    to_compute_grad = 0.
                    for k in range(n_bin):
                        to_compute_grad += torch.sqrt(torch.trace(loss_matrix[0,k]))

                    # Computing gradient for each energy bins (using DPS approximation).
                    grad = torch.autograd.grad(outputs=to_compute_grad, inputs=x_tdps, retain_graph=True, grad_outputs=torch.ones_like(to_compute_grad))
                    grad = torch.stack(list(grad), dim=0)[0,:,:]

                    # Gradient normalisation.
                    norm = torch.zeros_like(x_tdps).to(device)
                    for k in range(n_bin):
                        norm[0,k]=torch.linalg.norm(grad[0,k],ord='fro')


                    # Unconditional reverse diffusion (only once per iteration).
                    if step_k==0:
                        x_tdps = reverse_diffusion(x_tdps, t, alpha[t], alpha_bar[t], sigma2[t], img_size, spectral_nn, device)


                    # Gradient descent (conditional guidance).
                    x_tdps = x_tdps - step_list_tdps[None,:,None,None]*grad/norm

                    # From standardized to LAC coefficients using mean and std computed on the training data.
                    x_tdps_LAC = x_tdps*std_spectral + mean_spectral
                    x_tdps_LAC[x_tdps_LAC<0] = 0

            # Clear some memory.
            del grad, to_compute_grad, x_0, x_0_LAC, radon_x_pred, A_x0
            torch.cuda.empty_cache()                            


            # STEP 2 : Computes pseudo inverse of M and applies it to X_lac in order to obtain X_mat
            x_tdps_mat = torch.linalg.lstsq(M, x_tdps_LAC[0].reshape(n_bin, img_size**2), driver="gels").solution
            x_tdps_mat[x_tdps_mat<0] = 0
            x_tdps_mat = x_tdps_mat.reshape(n_mat, img_size, img_size)
            x_tdps_mat = x_tdps_mat[None,:,:,:]
            x_tdps_mat_np= x_tdps_mat.detach().cpu().numpy()

            PSNR_tdps = np.zeros(n_mat)
            SSIM_tdps = np.zeros(n_mat)
            for m in range(n_mat):
                PSNR_tdps[m]  = peak_signal_noise_ratio(x_true_material_np[0,m],x_tdps_mat_np[0,m],data_range=x_true_material_np[0,m].max())
                SSIM_tdps[m] = structural_similarity(x_tdps_mat_np[0,m] ,x_true_material_np[0,m], data_range=x_true_material_np[0,m].max(), gradient=False)

            # Saving .npy files
            file_path = f'res/{n}/{I0}/{n_iter}'
            if not os.path.exists(file_path): 
                os.makedirs(file_path) 
            np.save(f'{file_path}/TDPS.npy', x_tdps_mat_np)
            np.save(f'{file_path}/PSNR_TDPS.npy', PSNR_tdps)
            np.save(f'{file_path}/SSIM_TDPS.npy', SSIM_tdps) 



# # #                 ##############################################################
# # #                 # One-step material decomp with diffusion posterior sampling #
# # #                 #                      Simple U-Net version                  #
# # #                 ##############################################################

            x_odps = (x_DTV_md-mean_material)/std_material

            # Diffusion of initialisation to time nmax.
            x_odps = diffusion(x_odps, alpha_bar[n_iter])

            for t in range(n_iter-1,0,-1):
                for step_k in range(number_grad_descent):

                    with torch.enable_grad():
                        x_odps = x_odps.requires_grad_()

                        # Prediction of clean image x_0 from noisy image x_t (using tweedie's formula and score approximation).
                        x_0 = x_pred(x_odps, t, alpha_bar[t], img_size, material_nn, device).to(device)

                        x_0_mat = x_0*std_material + mean_material
                        x_0_mat[x_0_mat<0] = 0

                        x_0_LAC = M @ x_0_mat.squeeze().reshape(n_mat,img_size**2)
                        x_0_LAC = x_0_LAC.reshape(n_bin, img_size, img_size)[None,:,:,:]

                        x_0_LAC[x_0_LAC<0] = 0

                        # Projection of x_0_LAC according to the forward model.
                        radon_x_pred = radon.forward(x_0_LAC).to(device)
                        A_x0 = I*torch.exp(-radon_x_pred) + background

                    #  Data attachement term.
                    loss_matrix = torch.matmul(torch.transpose((measures-A_x0), 2, 3),(measures-A_x0)/(2*measures))
                    to_compute_grad = 0.
                    for k in range(n_bin):
                        to_compute_grad += torch.sqrt(torch.trace(loss_matrix[0,k]))

                    # Computing gradient for each energy bins (using DPS approximation).
                    grad = torch.autograd.grad(outputs=to_compute_grad, inputs=x_odps, retain_graph=True, grad_outputs=torch.ones_like(to_compute_grad))
                    grad = torch.stack(list(grad), dim=0)[0,:,:]

                    # Gradient normalisation.
                    norm = torch.zeros_like(x_odps).to(device)
                    for k in range(n_mat):
                        norm[0,k]=torch.linalg.norm(grad[0,k],ord='fro')


                    # Unconditional reverse diffusion (only once per iteration).
                    if step_k==0:
                        x_odps = reverse_diffusion(x_odps, t, alpha[t], alpha_bar[t], sigma2[t], img_size, material_nn, device)


                    # Gradient descent (conditional guidance).
                    x_odps = x_odps - step_list_odps[None,:,None,None]*grad/norm

            # Un-standardisation
            x_odps= x_odps*std_material + mean_material
            x_odps[x_odps<0] = 0                
            x_odps_np = x_odps.detach().cpu().numpy()

            PSNR_odps = np.zeros(n_mat)
            SSIM_odps = np.zeros(n_mat)
            for m in range(n_mat):
                PSNR_odps[m]  = peak_signal_noise_ratio(x_true_material_np[0,m],x_odps_np[0,m],data_range=x_true_material_np[0,m].max())
                SSIM_odps[m] = structural_similarity(x_odps_np[0,m] ,x_true_material_np[0,m], data_range=x_true_material_np[0,m].max(), gradient=False)

            # Saving .npy files
            file_path = f'res/{n}/{I0}/{n_iter}'
            if not os.path.exists(file_path): 
                os.makedirs(file_path) 
            np.save(f'{file_path}/ODPS.npy', x_odps_np)
            np.save(f'{file_path}/PSNR_ODPS.npy', PSNR_odps)
            np.save(f'{file_path}/SSIM_ODPS.npy', SSIM_odps) 


