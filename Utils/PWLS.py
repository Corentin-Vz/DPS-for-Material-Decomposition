import torch 
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
device = "cuda" if torch.cuda.is_available() else "cpu"

# Penalized WLS

# Regularization functions :
def quadr(x):
    psi0 = 0.5*torch.square(x)
    psi1 = x
    return psi0, psi1

def huber(x, delta):
    psi0 = torch.where(torch.abs(x)<=delta,0.5*torch.square(x),delta*torch.abs(x)-0.5*delta**2)
    psi1 = torch.where(torch.abs(x)<=delta,x,torch.sign(x)*delta)
    return psi0, psi1



# compute penalty value f and its gradient g
def imagePrior_huber(im,delta):
    
    # To create neighborhood
    l = torch.tensor([0,1,-1], device=device)
    vect = torch.zeros((9,2))
    
    ii = 0
    for i in range(3) :
        for j in range(3) :
            vect[ii,:] = torch.tensor([l[i],l[j]], device=device)
            ii = ii+1

    vect = vect[1:-1,:]
    F = 0
    D = 0
    
    for i in range(vect.size(dim=0)) :
        shift_vect = vect[i,:].int().tolist()
        im_shifted = torch.roll(torch.roll(im, shift_vect[0], dims=0), shift_vect[1], dims=1)
        
        # inverse of euclidian distance
        omega = torch.ones(im.shape, device=device)/np.linalg.norm(shift_vect)
        
        psi0, psi1 = huber(im-im_shifted, delta)
        
        F = F + omega*psi0
        D = D + 2*omega*psi1 
              
    f = torch.sum(F)
    g = torch.flatten(torch.transpose(D,0,1))
    
    return f, g

def diff_operator(x, weights, sign):
    # To create neighborhood
    l = torch.tensor([0,1,-1], device=device)
    vect = torch.zeros((9,2))
    
    ii = -1
    for i in range(3) :
        for j in range(3) :
            ii = ii+1
            vect[ii,:] = torch.tensor([l[i],l[j]], device=device)
            
    vect = vect[1:,:]

    res_shape = list(x.shape)
    res_shape.append(8)
    res = torch.zeros(res_shape, device=device)
    for i in range(vect.size(dim=0)) :
        shift_vect = vect[i,:].int().tolist()

        x_shifted = torch.roll(torch.roll(x, shift_vect[0], dims=0), shift_vect[1], dims=1).to(device)
        
        if weights == 'sqrt':
        # inverse of euclidian distance
            omega = torch.ones(x.shape, device=device)/np.linalg.norm(shift_vect)
            omega = torch.sqrt(omega)
        elif weights == 'standard':
            omega = torch.ones(x.shape, device=device)/np.linalg.norm(shift_vect)
        
        res[:,:,:,:,i] = (x+sign*x_shifted)*omega

    return res

def diff_operator_back(y, sign):
    # To create neighborhood
    l = torch.tensor([0,1,-1], device=device)
    vect = torch.zeros((9,2))
    
    ii = -1
    for i in range(3) :
        for j in range(3) :
            ii = ii+1
            vect[ii,:] = torch.tensor([l[i],l[j]], device=device)
            
    vect = vect[1:,:]

    res_shape = y.shape[0],y.shape[1],y.shape[2], y.shape[3]    
    res = torch.zeros(res_shape, device=device)
    for i in range(vect.size(dim=0)) :
        shift_vect = vect[i,:].int().tolist()

        y_shifted = torch.roll(torch.roll(y[:,:,:,:,i], -shift_vect[0], dims=0), -shift_vect[1], dims=1).to(device)
        
        # inverse of euclidian distance
        omega = torch.ones(res.shape, device=device)/np.linalg.norm(shift_vect)

        res += (y[:,:,:,:,i]+sign*y_shifted)*torch.sqrt(omega)

    return res

def PWLS(Mat, Spect, Measures, x_scout, n_iter, delta, beta_prior, device):

    x_mass_densities_np = Mat.x_mass_densities.detach().cpu().numpy()
    
    # From scout material images, compute pseudo spectral images (one for each bin) :
    x_pwls = Mat.mass_attn_pseudo_spectral @ x_scout.reshape(Mat.n_mat, Mat.img_size**2)
    x_pwls = x_pwls.reshape(1, Spect.n_bin, Mat.img_size, Mat.img_size)
    
    # Computing an approximation of the n_bin sinograms from the measures :
    sino_approx_spectral = torch.log(torch.sum(Spect.binned_spectrum, dim=1)[:,None,None]/Measures.y)
    sino_approx_spectral[Measures.y==0] = 0
    sino_approx_spectral = sino_approx_spectral[None]
    
    # Weights
    w = (Measures.y-Measures.background)**2
    w[Measures.y-Measures.background<=0] = 0
    D = Measures.radon.backward(w*Measures.radon.forward(torch.ones_like(x_pwls)))

    beta_prior = beta_prior[None,:,None,None]

    PSNR_pwls = np.zeros([n_iter, Mat.n_mat])
    SSIM_pwls = np.zeros([n_iter, Mat.n_mat])

    for n in range(n_iter):
        x_rec = x_pwls-Measures.radon.backward(w*(Measures.radon.forward(x_pwls)-sino_approx_spectral))/D

        diff_x = diff_operator(x_pwls, 'standard', -1)
        psi0, psi1 = huber(diff_x, delta)

        c = psi1/diff_x
        c[diff_x==0] = 1

        D_reg = diff_operator_back(c*diff_operator(torch.ones_like(x_pwls, device=device),'sqrt',1),1)
        x_reg = x_pwls - diff_operator_back(c*diff_operator(x_pwls,'sqrt',-1),-1)/D_reg

        x_pwls = (D*x_rec + beta_prior*D_reg*x_reg)/(D+ beta_prior*D_reg)
        x_pwls[x_pwls<0]=0
        
        # Pseudo inverse :
        x_temp = torch.linalg.lstsq(Mat.mass_attn_pseudo_spectral, 
                                    x_pwls[0,:].reshape(Spect.n_bin, Mat.img_size**2),
                                    driver="gels").solution.reshape(2, Mat.img_size, Mat.img_size)[None,:,:,:]
        x_temp[x_temp<0]=0
        
        # Computing metrics :
        for k in range(Mat.n_mat):
            PSNR_pwls[n,k] = peak_signal_noise_ratio(x_mass_densities_np[0,k],x_temp[0,k].detach().cpu().numpy(),data_range=x_mass_densities_np[0,k].max())
            SSIM_pwls[n,k] = structural_similarity(x_temp[0,k].detach().cpu().numpy(), x_mass_densities_np[0,k], data_range=x_mass_densities_np[0,k].max(), gradient=False)


    return x_temp, x_pwls, PSNR_pwls, SSIM_pwls
