import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os 

class DataLoader_spectral(Dataset):

    def __init__(self, img_dir, energy_list, patient_list, transform):
        self.img_dir = img_dir
        self.energy_list = energy_list
        self.transform = transform
        self.patient_list = patient_list

        # Computing n_data = total number of slices
        patient_file_number = torch.zeros([len(patient_list)+1])
        for n in range(len(patient_list)):
            patient_file_number[n+1] = len(os.listdir(img_dir + f'{patient_list[n]}/{energy_list[0]}_kev/'))
        patient_total_file_number = torch.cumsum(patient_file_number,0)
        self.n_data=int(patient_total_file_number[-1].item())
        self.patient_total_file_number = patient_total_file_number
        self.patient_file_number = patient_file_number

    def __getitem__(self,index):
        imgs = np.zeros([512, 512, len(self.energy_list)])
        for n in range(1,len(self.patient_list)+1):
            if index >= self.n_data:
                return "index out of range"
            if index < self.patient_total_file_number[n]:
                b = 0
                for energy in self.energy_list:
                    file = self.img_dir+str(f'{self.patient_list[n-1]}/{energy}_kev/{index-int(self.patient_total_file_number[n-1])}.npy')
                    imgs[:,:,b] = np.load(file)
                    b +=1
                break
        
        if self.transform:
            imgs = self.transform(imgs)
        
        return imgs, index
    
    def __getpath__(self,index):
        imgs = np.zeros([512, 512, len(self.energy_list)])
        for n in range(1,len(self.patient_list)+1):
            if index >= self.n_data:
                return "index out of range"
            if index < self.patient_total_file_number[n]:
                file = []
                for energy in self.energy_list:
                    file.append(self.img_dir+str(f'{self.patient_list[n-1]}/{energy}_kev/{(index-int(self.patient_total_file_number[n-1])).item()}.npy'))

                break
        return file
    
    def __len__(self):
        return self.n_data
 


    
    

