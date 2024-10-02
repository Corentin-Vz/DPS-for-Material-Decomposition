import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

def loss_fn(model,x,T,alpha_bar):
    loss = torch.nn.MSELoss()
    z = torch.randn_like(x)
    t = torch.randint(T,(x.shape[0],), device=x.device) 
    x_noise = torch.sqrt(alpha_bar[t][:, None, None, None] )*x + torch.sqrt(1-alpha_bar[t][:, None, None, None] )*z

    z_estimate = model(x_noise,t)
      
    return loss(z, z_estimate)

def loss_fn_with_ip(model,x,T,alpha_bar, reg_param, mean, std):
    loss = torch.nn.MSELoss()
    z = torch.randn_like(x)
    t = torch.randint(T,(x.shape[0],), device=x.device) 
    x_noise = torch.sqrt(alpha_bar[t][:, None, None, None] )*x + torch.sqrt(1-alpha_bar[t][:, None, None, None] )*z
    z_estimate = model(x_noise,t)
    
    x_pred = (x - torch.sqrt(1-alpha_bar[t][:, None, None, None] )*z_estimate) /(torch.sqrt(alpha_bar[t])[:, None, None, None] ) 
    
#     x_pred_detached = x_pred.detach()
#     J = x_pred_detached.shape[-1]*x_pred_detached.shape[-2]
#     x_pred_detached = torch.flatten(x_pred_detached, start_dim=2, end_dim=-1)
#     x_sup = torch.max(torch.abs(x_pred_detached), dim=-1).values
#     ip_reg = reg_param*(torch.sum((x_pred[:,0,:,:]+x_sup[:,0][:,None,None])*(x_pred[:,1,:,:]+x_sup[:,1][:,None,None])))/J
    x_pred = std*(x_pred + mean)
    ip_reg = reg_param*(torch.mean(x_pred[:,0,:,:]*x_pred[:,1,:,:]))


    return loss(z, z_estimate) + ip_reg


def diffusion_parameters(T):
    beta = np.geomspace(4e-4,4e-2,T)
    beta = torch.from_numpy(beta)
    alpha = 1 - beta
    alpha_bar = torch.ones(T)
    alpha_bar[0] = alpha[0]
    for i in range(1,T) :
        alpha_bar[i] = alpha_bar[i-1]*alpha[i]
    return alpha, alpha_bar

def diffusion_parameters_linear(T):
    beta = np.geomspace(1e-4,2e-2,T)
    beta = torch.from_numpy(beta)
    alpha = 1 - beta
    alpha_bar = torch.ones(T)
    alpha_bar[0] = alpha[0]
    for i in range(1,T) :
        alpha_bar[i] = alpha_bar[i-1]*alpha[i]
    return alpha, alpha_bar

def diffusion(x, alpha_bar_t):
    # Diffuse image x to time t
    z = torch.randn_like(x, requires_grad = False)
    x_t = torch.sqrt(alpha_bar_t)*x+torch.sqrt(1-alpha_bar_t)*z
    return x_t

def reverse_diffusion(x, t, alpha_t, alpha_bar_t, sigma2_t, img_size, ddpm_model, device):
    b, c, h, w = x.size()
    t_tensor = (t)*torch.ones(b)
    
    # Prédiction du bruit z par le réseau de neuronnes :
    z_pred = ddpm_model(x.to(device), t_tensor.to(device)).detach()

    
    # Moyenne de la diffusion inverse
    mu = 1/torch.sqrt(alpha_t)*x
    mu = mu - (1-alpha_t) / (torch.sqrt(1-alpha_bar_t)) / torch.sqrt(alpha_t) * z_pred
        


    if t!=0:
#         x = mu + torch.sqrt(1-alpha_t)*torch.randn_like(x)
        # ??
        x = mu + torch.sqrt(sigma2_t) * torch.randn_like(x) 
        # ??
    else :
        x = mu
    
    return x

def reverse_diffusion_deterministic(x, t, alpha_t, alpha_bar_t, sigma2_t, img_size, ddpm_model, device):
    b, c, h, w = x.size()
    t_tensor = (t)*torch.ones(b)
    
    # Prédiction du bruit z par le réseau de neuronnes :
    z_pred = ddpm_model(x.to(device), t_tensor.to(device)).detach()/2.

    
    # Moyenne de la diffusion inverse
    mu = 1/torch.sqrt(alpha_t)*x
    mu = mu - (1-alpha_t) / (torch.sqrt(1-alpha_bar_t)) / torch.sqrt(alpha_t) * z_pred
    
    return mu

def x_pred(x, t, alpha_bar_t, img_size, ddpm_model, device):
    b, c, h, w = x.shape
    t_tensor = (t)*torch.ones(b)

    # Prédiction du bruit z par le réseau de neuronnes :
    z_pred = ddpm_model(x, t_tensor.to(device)).detach()

    if t!=0:
        x0 = (x - torch.sqrt(1-alpha_bar_t)*z_pred) /torch.sqrt(alpha_bar_t) 

    else :
        x0= x

    return x0

def reverse_diffusion_from_x_pred(x, x_0_hat, t, alpha_t, alpha_bar_t, sigma2_t, img_size, ddpm_model, device):
    
    mu = torch.sqrt(alpha_bar_t/alpha_t)*(1-alpha_t)/(1-alpha_bar_t)*x_0_hat + torch.sqrt(alpha_t)*(1-alpha_bar_t/alpha_t)/(1-alpha_bar_t)*x
    
    if t!=0:
        x_t = mu + torch.sqrt(sigma2_t)*torch.randn_like(x)
    
    else :
        x_t = mu
    
    return x_t



