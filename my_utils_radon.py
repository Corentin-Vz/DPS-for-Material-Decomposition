import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

air = {
'density':1.205e-3,
20:7.779e-1,
40:2.485e-1,
60:1.875e-1,
80:1.662e-1,
100:1.541e-1,
120:1.541e-1+(1.614e-1-1.541e-1)*2/5,
140:1.541e-1+(1.614e-1-1.541e-1)*4/5
}
water = {
'density':1e-0,
20:8.096e-1,
40:2.683e-1,
60:2.059e-1,
80:1.837e-1,
100:1.707e-1,
120:1.707e-1+(1.505e-1-1.707e-1)*2/5,
140:1.707e-1+(1.505e-1-1.707e-1)*4/5
}



def get_filter(size, sinogram):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * torch.fft.rfft(torch.from_numpy(f))  # ramp filter

    return fourier_filter

def filter_sinogram(sinogram, filter_name="ramp"):
    # Sinogram has shape [batch size, channel, detector size, n_angles]
    size = sinogram.size(-1)
    n_angles = sinogram.size(-2)

    # Pad sinogram to improve accuracy
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
    pad = 0
    padded_sinogram = F.pad(sinogram.float(), (0, pad, 0, 0))

    sino_fft = torch.fft.rfft(sinogram) 

    
    # get filter and apply
    f = get_filter(size, sinogram.device)
    filtered_sino_fft = sino_fft * f.cuda()

    # Inverse fft
    filtered_sinogram = torch.fft.irfft(filtered_sino_fft)
    filtered_sinogram = filtered_sinogram[:,:,:, :] * (np.pi / (2 * n_angles))

    return filtered_sinogram.to(dtype=sinogram.dtype)

def get_HU_array(array, energy, spacing):
    HU_array = 1000 * (spacing*array-water[energy]*water['density']) / (water[energy]*water['density'] - air[energy]*air['density'])
    return HU_array
