import pandas
import numpy as np
import torch
from torch_radon import Radon, RadonFanbeam

def create_bins(spectral_incident, bin_list, prop_factor, device):
    '''
    Create a binned spectrum. No cross detection.
    
    INPUTS :
    Spectral_incident : spectral incident intensity array.
    bin_list : list of energy intervals.
    prop_factor : scalar to multiply by in order to alter the total number of photon.
    
    OUTPUTS :
    n_bin : int. Number of bins.
    h_k : [n_bin, 150] ndarray.
    E_k : [n_bin] array. Mean energy of each bin.
    S_k : [n_bin] array. Sum of photon count for each bin.
    
    '''
    h_k = torch.zeros(len(bin_list)-1, spectral_incident.shape[0], device=device)
    for k in range(len(bin_list)-1):
        bin_k = torch.zeros(spectral_incident.shape[-1], device=device)
        bin_k[bin_list[k]:bin_list[k+1]] = 1
        h_k[k,:] = spectral_incident*bin_k
        
    h_k = prop_factor * h_k
    n_bin = len(bin_list)-1
    S_k = torch.sum(h_k, dim=1)
    E_k = torch.round(torch.sum(torch.arange(150, device=device)*h_k, dim=1)/S_k).long()

    return n_bin, prop_factor * h_k, E_k, S_k
 
def create_mass_attenuation_matrix(E_k, x_true_mat, pixel_size, device):
    '''
    Create the mass attenuation matrix for bones and soft tissues (n_mat=2).
    
    INPUT : 
    E_k. [n_bin] array. Mean of each energy bin.
    
    OUPUTS :
    Q.                 [150, n_mat] tensor.                    Contains all mass attenuation coefficients of the materials for all 150 energies.
    Q_speudo_spectral. [n_bin, n_mat] tensor.                  Contains mass att coeff but only for the mean energy of each bin.
    x_mass_density.    [n_mat, pixel_size, pixel_size] tensor. Material images in g.cm^(-3).
    '''
    
    # Loading the material mass attenuation data
    df = pandas.read_csv('csv_files/mass_att.csv') # mm^-1
    rho_df = pandas.read_csv('csv_files/rho.csv')  # g cm^(-3)
    rho = torch.tensor([rho_df['Soft Tissues'][0] , rho_df['Bones'][0] ])
    
    Q = torch.zeros(150,2, device=device)
    Q[:,1] = torch.tensor(df['Soft Tissues']  / rho_df['Soft Tissues'][0] ) * 10 # in cm^2 g^(-1)
    Q[:,0] = torch.tensor(df['Bones']       / rho_df['Bones'][0] ) * 10          # in cm^2 g^(-1)
    


    n_bin = len(E_k)
    Q_pseudo_spectral = torch.zeros(n_bin,2, device=device)
    Q_pseudo_spectral[:,0] = Q[E_k,0]
    Q_pseudo_spectral[:,1] = Q[E_k,1]
    
    
    x_mass_density = torch.zeros_like(x_true_mat)
    x_mass_density[0,0] = rho_df['Bones'][0]        * x_true_mat[0,0] * pixel_size  # in g.cm^(-3)
    x_mass_density[0,1] = rho_df['Soft Tissues'][0] * x_true_mat[0,1] * pixel_size  # in g.cm^(-3)

    
    return Q, Q_pseudo_spectral, x_mass_density, rho

def create_radon_op(img_size, n_angles, max_angle, geom, device):
    angles = np.linspace(0, max_angle, n_angles, endpoint=False)
    det_width = 1.2
    det_count = 750 
    
    if geom =='parallel':
        radon = Radon(resolution=img_size, angles=angles, det_count=det_count)
    
    elif geom == 'fanbeam':
        source_origin =  600
        origin_det = 600
        radon = RadonFanbeam(resolution=img_size, angles=angles,det_count= det_count, 
                             det_spacing=det_width, source_distance=source_origin, 
                             det_distance=origin_det, clip_to_circle=False)
    return radon

def forward_mat_op(x_mass_density, Q, S, background, radon, noise, device):
    '''
    Returns simulated measures following the forward model : 
    Y ~ Poisson( S exp(-QRadon(X)) + background )
    
    INPUTS :
    x_mass_density [1, n_mat, img_size, img_size] tensor.  Material density images.
    Q              [150, n_mat] tensor.                    Mass Att matrix.
    S              [n_bin, 150] tensor.                    Binned spectrum from create_bins function.
    background     scalar.                                 Dark current.
    radon          TorchRadon                              From create_radon_op function.
    noise          Boolean.                                If True, applies poisson noise. Else, the expected mean Y_bar is returned.
    
    OUTPUT :
    Y or Y_bar. [n_bin, n_angles, det_count] tensor. 
    '''
    batch, n_mat, img_size, no = x_mass_density.shape
    n_bin = S.shape[0]
    
    x_proj = radon.forward(x_mass_density)                                 # Radon(X)
    Qx_proj = Q @ x_proj.reshape(n_mat, x_proj.shape[-1]*x_proj.shape[-2]) # QRadon(X)
    Total_att = torch.exp(-Qx_proj)                                        # exp(-Qradon(X))
    Y_bar = S @ Total_att                                                  # S exp(-Qradon(X))
    Y_bar += background                                                    # S exp(-Qradon(X)) + background
    if noise :
        Y = torch.poisson(Y_bar)
        return Y.reshape(n_bin, x_proj.shape[-2], x_proj.shape[-1])
    else :
        return Y_bar.reshape(n_bin, x_proj.shape[-2], x_proj.shape[-1])
