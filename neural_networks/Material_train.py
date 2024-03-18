import sys
sys.path.append("/share/castor/home/vazia/DPS")
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from Data.DataLoader_material import DataLoader_material
from torch.optim import Adam
from my_utils_ddpm import loss_fn, diffusion_parameters
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# CHECKLIST :
# - chkpt name (checkpoint name to save the weights of the NN)
# - lr (learning rate and scheluder)
# - epochs
# - logdir (for tensorboard if needed)
# - batch_size


# Transforms to apply to images (resize, data augmentation, ..)      
my_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5)] 
)                                   

# Train dataset
material_list = ['Bones', 'Soft Tissues']
patient_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
patient_list_val = [10]
# (and patient_test = 12)


batch_size = 32
n_mat = len(material_list)
data_train = DataLoader_material(img_dir ='../Data/', material_list=material_list, patient_list=patient_list, transform=my_transforms)
data_val = DataLoader_material(img_dir ='../Data/', material_list=material_list, patient_list=patient_list_val, transform=my_transforms)

Data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
Data_loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=True)

mean = torch.tensor(np.load('../Data/mean_material.npy'))[None,:,None,None]
std = torch.tensor(np.load('../Data/std_material.npy'))[None,:,None,None]

# U-NET
from UNet import UNet
ddpm_model = UNet(image_channels = n_mat, n_channels=8)
ddpm_model = ddpm_model.to(device)
ckpt_name = "../checkpoints_material/nn_weights/material.pth"
# If poursuing a previous training :
# ddpm_model.load_state_dict(torch.load(ckpt_name))


# # Optimisation parameters
lr = 1e-3
n_epochs = 400

optimizer = Adam(ddpm_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
writer = SummaryWriter(log_dir="../checkpoints_material/runs/")

# # DIFFUSION PARAMETERS & FUNCTIONS

T = 1000
alpha, alpha_bar = diffusion_parameters(T)
alpha_bar = alpha_bar.to(device).requires_grad_(False)

for epoch in range(n_epochs):

    avg_loss = 0.
    num_items = 0
    
    avg_loss_val = 0.
    num_items_val = 0
    
    
    for x, index in Data_loader_train:

        x = (x-mean)/std
        x = x.float()

        x = x.to(device).requires_grad_(False)
        loss = loss_fn(ddpm_model, x, T, alpha_bar)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
     

        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    with torch.no_grad():
        for x_val, index in Data_loader_val:

            x_val = (x_val-mean)/std
            x_val = x_val.float()
            x_val = x_val.to(device).detach().requires_grad_(False)
            
            loss = loss_fn(ddpm_model, x_val, T, alpha_bar)

            avg_loss_val += loss.item() * x_val.shape[0]
            num_items_val += x_val.shape[0]
            
    writer.add_scalars('run', {'Train': avg_loss/num_items,
                                'Val': avg_loss_val/ num_items_val}, epoch)    
    scheduler.step()

    if epoch%10==0:
        torch.save(ddpm_model.state_dict(), ckpt_name)

# Update the checkpoint at the end of training.
torch.save(ddpm_model.state_dict(), ckpt_name)
writer.flush()



