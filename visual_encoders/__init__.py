import os
import json

import numpy as np
import torch

from visual_encoders.dlp.models import ObjectDLP
from visual_encoders.dlp.utils.util_func import plot_keypoints_on_image
from visual_encoders.vae.models.vqvae import VQModel


def load_pretrained_rep_model(cfg):
    
	if cfg.obs in ['state', 'ec_state', 'ec_state_gen', 'rgb']:
		return

	elif cfg.obs == 'dlp':
		print("\nLoading pretrained DLP...")
		# load config
		conf_path = os.path.join(cfg.rep_model_checkpoint, 'hparams.json')
		with open(conf_path, 'r') as f:
			config = json.load(f)
		# initialize model
		model = ObjectDLP(cdim=config['cdim'], enc_channels=config['enc_channels'],
							prior_channels=config['prior_channels'],
							image_size=config['image_size'], n_kp=config['n_kp'],
							learned_feature_dim=config['learned_feature_dim'],
							bg_learned_feature_dim=config['bg_learned_feature_dim'],
							pad_mode=config['pad_mode'],
							sigma=config['sigma'],
							dropout=False, patch_size=config['patch_size'], n_kp_enc=config['n_kp_enc'],
							n_kp_prior=config['n_kp_prior'], kp_range=config['kp_range'],
							kp_activation=config['kp_activation'],
							anchor_s=config['anchor_s'],
							use_resblock=False,
							scale_std=config['scale_std'],
							offset_std=config['offset_std'], obj_on_alpha=config['obj_on_alpha'],
							obj_on_beta=config['obj_on_beta'])
		# load model from checkpoint
		ckpt_path = os.path.join(cfg.rep_model_checkpoint, 'best.pth')
		model.load_state_dict(torch.load(ckpt_path))
		# set number of encoded keypoints post-training (optional)
		if cfg.dlp_post_training_num_kp > 0:
			# NOTE: n_kp_enc must be smaller than n_kp_prior, otherwise there is no filtering of particles
			model.fg_module.n_kp_enc = cfg.dlp_post_training_num_kp
			print(f"Number of encoded particles set to {model.fg_module.n_kp_enc} post-training")

	elif cfg.obs == 'vqvae':
		print("\nLoading pretrained VQ-VAE...")
		# load config
		conf_path = os.path.join(cfg.rep_model_checkpoint, 'hparams.json')
		with open(conf_path, 'r') as f:
			config = json.load(f)
        # initialize model
		model = VQModel(embed_dim=config['embed_dim'],
                        n_embed=config['n_embed'],
                        double_z=False,
                        z_channels=config['z_channels'],
                        resolution=config['image_size'],
                        in_channels=config['ch'],
                        out_ch=config['ch'],
                        ch=config['base_ch'],
                        ch_mult=config['ch_mult'],  # num_down = len(ch_mult)-1
                        num_res_blocks=config['num_res_blocks'],
                        attn_resolutions=config['attn_resolutions'],
                        dropout=config['dropout'],
                        device=torch.device(config['device']),
                        ckpt_path=config['pretrained_path'],
                        ignore_keys=[],
                        remap=None,
                        sane_index_shape=False)
		# load model from checkpoint
		ckpt_path = os.path.join(cfg.rep_model_checkpoint, 'best.pth')
		model.load_state_dict(torch.load(ckpt_path))
	
	elif cfg.obs == 'vae':
		raise NotImplementedError

	else:
		raise NotImplementedError(f"Pretrained model type '{cfg.obs}' is not supported")

	model.eval()
	model.requires_grad_(False)
	model.to(torch.device(cfg.rep_model_device))

	print(f"Loaded pretrained representation model from {ckpt_path}\n")

	return model


"""
DLP Related
"""

def get_dlp_rep(dlp_output):
    pixel_xy = dlp_output['z']
    scale_xy = dlp_output['mu_scale']
    depth = dlp_output['mu_depth']
    transp = dlp_output['obj_on'].unsqueeze(dim=-1)
    visual_features = dlp_output['mu_features']
    rep = torch.cat((pixel_xy, scale_xy, depth, transp, visual_features), dim=-1)
    return rep


def extract_dlp_features(images, dlp_model):
    normalized_images = images.permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255

    with torch.no_grad():
        encoded_output = dlp_model.encode_all(normalized_images, deterministic=True)
        particles = get_dlp_rep(encoded_output)

    return particles #, encoded_output['z_bg']


def extract_dlp_image(images, obs_mean, obs_std, latent_rep_model, device='cuda:0'):
	orig_image_shape = images.shape
	if len(orig_image_shape) == 3:
		images = np.expand_dims(images, axis=0)
	normalized_images = images.astype('float32') / 255
	normalized_images = torch.tensor(normalized_images, device=torch.device(device)).contiguous()

	with torch.no_grad():
		encoded_output = latent_rep_model.encode_all(normalized_images, deterministic=True)
		pixel_xy = encoded_output['z']
		obj_on = encoded_output['obj_on']

	normalized_obs = (get_dlp_rep(encoded_output) - obs_mean.to(device)) / obs_std.to(device)
	normalized_obj_on = normalized_obs[:, :, 5]

	dlp_images = []
	for kp_xy, kp_on, kp_on_norm, image in zip(pixel_xy, obj_on, normalized_obj_on, normalized_images):
		# kp_xy = kp_xy[kp_on > 0.98]  # uncomment to filter out based on transparency value
		# kp_xy = kp_xy[kp_on_norm > 0]  # uncomment to filter out based on normalized transparency value
		dlp_images.append(plot_keypoints_on_image(kp_xy, image, radius=1, thickness=1, kp_range=(-1, 1), plot_numbers=False))

	if len(dlp_images) == 1:
		dlp_images = dlp_images[0]

	return dlp_images


"""
VQ-VAE Related
"""

def extract_vqvae_features(images, vqvae_model):
	normalized_images = images.permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255

	with torch.no_grad():
		normalized_image = vqvae_model.preprocess_rgb(normalized_images)
		vqvae_features = vqvae_model.get_latent_rep(normalized_image, deterministic=True)

	vqvae_features = vqvae_features.reshape(vqvae_features.shape[0], -1)  # flatten
	return vqvae_features
