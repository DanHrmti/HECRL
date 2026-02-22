#!/bin/bash


#### npz to dir format ####

# In order to train a visual encoder, you first need to convert the npz dataset to a directory format.
# You must insert the data_dir you want to save the processed dataset to and the path to the source npz dataset
# to the following script:

python visual_encoders/npz_to_dir_format.py

###########################


#### DLP ####

# Set configuration file in visual_encoders/dlp/config_dlp.yaml
# You must insert the 'data_root_dir'.
# To reproduce the DLP checkpoints used in the paper, use the following values for 'n_kp_enc': 20 for PPP-Cube, 24 for Scene, 20 for Push-Tetris
# All other values are set to the ones used in the paper across all environments.

# multi-GPU training
accelerate launch --config_file visual_encoders/dlp/accel_conf.yml visual_encoders/dlp/train_dlp_accelerate.py  # see top of visual_encoders/dlp/train_dlp_accelerate.py script for instructions

# single-GPU training
python visual_encoders/dlp/train_dlp.py  # see top of visual_encoders/dlp/train_dlp.py script for instructions

#############


#### VQ-VAE ####

# Each environment/dataset has its own config file which can be found in visual_encoders/vae/configs
# Set conf_path variable directly in visual_encoders/vae/train_vae.py

# single-GPU training
python visual_encoders/vae/train_vae.py

################

