# Python code for One-step and Two-step Material Decomposition (ODPS and TDPS) using diffusion models.
# Data Structure : 
The folder "Data" must contain folders of patients ("0", "1", ...). Inside each patient folder : energy bins folders ("40_kev", "80_kev", ... ) and material folders ("Bones", "Soft Tissues"). Then, each subfolder contains slices of the patient ("0.npy", "1.npy", ...).
*Example :
"Data/2/60_kev/10.npy" ; "Data/2/80_kev/10.npy" ; "Data/2/100_kev/10.npy" --> 10th slice of patient number 2 at energy 60, 80 and 100 keV. 
Note that the corresponding material image matches, i.e "Data/2/Bones/10.npy" ; "Data/2/Soft Tissues/10.npy" are the corresponding material images.*

Energy bins and material can differ from this, in this case the variables "energy_list" and "material_list" must be changed accordingly in the training files.

If attenuation images are not in pixel^(-1), the variable "pixel_size" must be changed accordingly, for both training and material decomposition.

We also need to compute the mean and standard deviation (std) of the training data. Std can be computed with the formula sqrt(E[x²] - E[X]²), this way we can cycle throught the data "one slice at a time". They are computed per energy bins/materials.
Mean and std are used for preprocessing the input image(s) x_in of the neural network (NN) x_in <-- (x_data-mean)/std and for post processing the output x_out of the NN x <-- std*x_out + mean.

# Files :

"torch_radon_env_save.yml" : A conda save of the enviroment and packages used. 

Note that currently, "arguments" are **not** python args and need to be changed within the .py file manually.

"Spectral_train.py" takes for arguments:
- "patient_list" : List of patient numbers we want to use. *e.g "patient_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]" for training and "patient_list_val =[10]" for validation.* 
- "energy_list" : List of selected energy bins. *e.g "energy_list = [40,80,120]" for 40, 80 and 120 keV.*
- "pixel_size" : equals to 1 if data attenuation images are already in pixel^(-1).
- "mean_spectral" and "sd_spectral" : mean and standard deviation computed on the training dataset
- deep learning arguments : epochs, batch_size, learning rate ...

  
"Material_train.py" takes for arguments:
- "patient_list" : List of patient numbers we want to use. *e.g "patient_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]" for training and "patient_list_val =[10]" for validation. *
- "material_list" : List of selected material folder names. *e.g "material_list = ['Bones', 'Soft Tissues]".*
- "mean_material" and "sd_material" : mean and standard deviation computed on the training dataset
- deep learning arguments : epochs, batch_size, learning rate ...

"test_material.py" : This is the script for testing both TDPS and ODPS along side with DTV. A few notes : Right now, weighted least square (WLS) is obtained using the DTV code with a dummy reference image and regularization parameter beta = 0.
**This script will be factorized into multiple python files, one for each method. WIP.**
