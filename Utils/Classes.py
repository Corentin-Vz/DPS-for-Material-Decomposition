import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from dataclasses import dataclass
import numpy as np


from Utils.forward_model import create_mass_attenuation_matrix, create_bins
import pandas
@dataclass
class SpectrumClass:
    def __init__(self, bin_list, prop_factor, device):
        '''
        S : Full spectrum. [150] array
        bin_list : list of energy intervals.
        prop_factor : scalar to multiply by in order to alter the total number of photon.
        n_bin : int. Number of bins.
        binned_spectrum : [n_bin, 150] ndarray.
        mean_energies : [n_bin] array. Mean energy of each bin.
        sum_over_bins : [n_bin] array. Sum of photon count for each bin.
        '''
        self.S = torch.tensor(pandas.read_csv('csv_files/spectral_incident.csv')['120_keV']).to(device)
        self.bin_list = bin_list
        self.prop_factor = prop_factor
        self.n_bin, self.binned_spectrum, self.mean_energies, self.sum_over_bins = create_bins(self.S,
                                                                                               bin_list,
                                                                                               prop_factor,
                                                                                               device)
        
class MaterialClass:
    def __init__(self, x_data, pixel_size, Spect, device):
        self.pixel_size = pixel_size
        self.img_size = x_data.shape[-1]
        self.n_mat = x_data.shape[1]
        self.mass_attn, self.mass_attn_pseudo_spectral, self.x_mass_densities, self.rho = create_mass_attenuation_matrix(Spect.mean_energies,
                                                                                                                    x_data,
                                                                                                                    pixel_size,
                                                                                                                    device)
from Utils.forward_model import forward_mat_op, create_radon_op
class MeasureClass:
    
    def __init__(self, Mat, Spect, sino_shape, max_angle, geom, background, device):
        self.n_angles, self.det_count = sino_shape
        self.max_angle = max_angle
        self.background = background
        
        if geom == 'parallel':
            self.radon = create_radon_op(img_size=Mat.img_size,
                                        n_angles=self.n_angles,
                                        max_angle=self.max_angle,
                                        det_count=self.det_count,
                                        geom='parallel',
                                        device=device)
            
        elif geom == 'fanbeam':
            self.radon = create_radon_op(img_size=Mat.img_size,
                                        n_angles=self.n_angles,
                                        max_angle=self.max_angle,
                                        det_count=self.det_count,
                                        geom='fanbeam',
                                        device=device)
        
        self.y = forward_mat_op(Mat.x_mass_densities,
                                Mat.mass_attn,
                                Spect.binned_spectrum,
                                background,
                                self.radon,
                                True,
                                device)
        

from Utils.my_utils_ddpm import diffusion_parameters 
from neural_networks.UNet import UNet
class ODPSClass:
    '''
    nn : neural network
    mean, std : mean and standard deviation used to standardize images before getting in the nn. They were computed on training data.
    T, alpha, alpha_bar, sigma2 :  Diffusion paramters (obtained with diffusion_parameters function).
    '''
    def __init__(self):
        self.mean = torch.tensor(np.load('Data/mean_material.npy'), device=device, requires_grad=False)[None,:,None,None]
        self.std = torch.tensor(np.load('Data/std_material.npy'), device=device, requires_grad=False)[None,:,None,None]
        self.nn = UNet(image_channels = 2, n_channels=32, n_blocks=2)
        self.nn = self.nn.to(device)
        ckpt_name = f"checkpoints_material/nn_weights/material_sept.pth"
        self.nn.load_state_dict(torch.load(ckpt_name))
        self.T = 1000
        self.alpha, self.alpha_bar = diffusion_parameters(self.T)
        self.sigma2 = (1-self.alpha)*(1-self.alpha_bar/self.alpha)/(1-self.alpha_bar)

class TDPSClass:
    '''
    nn : neural network
    mean, std : mean and standard deviation used to standardize images before getting in the nn. They were computed on training data.
    T, alpha, alpha_bar, sigma2 :  Diffusion paramters (obtained with diffusion_parameters function).
    '''
    def __init__(self):
        self.mean = torch.tensor(np.load('Data/mean_pseudo_spectral.npy'), device=device, requires_grad=False)[None,:,None,None]
        self.std = torch.tensor(np.load('Data/mean_pseudo_spectral.npy'), device=device, requires_grad=False)[None,:,None,None]
        self.nn = UNet(image_channels = 3, n_channels=32, n_blocks=2)
        self.nn = self.nn.to(device)
        ckpt_name =  f"checkpoints/nn_weights/pseudo_spectral2.pth"
        self.nn.load_state_dict(torch.load(ckpt_name))
        self.T = 1000
        self.alpha, self.alpha_bar = diffusion_parameters(self.T)
        self.sigma2 = (1-self.alpha)*(1-self.alpha_bar/self.alpha)/(1-self.alpha_bar)

