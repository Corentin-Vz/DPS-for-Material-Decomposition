#!/bin/bash -l
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc

# On peut éventuellement placer ici les commentaires SBATCH permettant de définir les paramètres par défaut de lancement :
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=8g
#SBATCH -p longrun
#SBATCH -t 3-00:00:00 
#SBATCH -C m48
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=corentin.vazia@gmail.com


conda activate myenv

# Exécution du script habituellement utilisé, on utilise la variable CUDA_VISIBLE_DEVICES qui contient la liste des GPU logiques actuellement réservés (toujours à partir de 0)

python3 Material_unet_train.py
