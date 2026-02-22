import os
from tqdm import trange
from copy import deepcopy

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from common.buffer import Buffer


def get_filename(cfg, validation):
    filename = f'{cfg.train_dataset_name}'
    
    if cfg.obs in ['dlp', 'vqvae']:
        filename = filename + '-' + f'{cfg.obs}'
    
    if validation:
        filename = filename + '-' + 'val'
    
    return filename


def load_dataset_from_file(dataset_path, cfg):
    file = np.load(dataset_path)
    
    dataset = dict()

    for k in file.keys():
        if k == 'observations':
            if cfg.obs in ['state', 'ec_state', 'ec_state_gen']:
                file_key = 'state_observations'
                dtype = np.float32
            else:
                file_key = 'observations'
                if 'dlp' in dataset_path or 'vqvae' in dataset_path:
                    dtype = np.float32
                else:
                    dtype = np.uint8
        else:
            file_key = k
            if 'image' in k:
                dtype = np.uint8
            else:
                dtype = np.float32
        dataset[k] = file[file_key][...].astype(dtype, copy=False)
        dataset[k] = dataset[k][:dataset[k].shape[0] // cfg.train_dataset_div]  # NOTE: optional reduction of dataset size

    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)

    return dataset


def preprocess_and_save_dataset(cfg, env, save_path):

    fp = os.path.join(cfg.data_dir, f'{cfg.train_dataset_name.replace(f"-{cfg.obs}", "")}.npz')
    print(f"Loading dataset from {fp}...")
    np_dataset = np.load(fp)

    obs = torch.from_numpy(np_dataset["observations"])
    preprocess_info = deepcopy(env.preprocess_info)
    
    observations = []
    chunk_size = 500
    for i in trange(0, len(obs), chunk_size):
        batch_obs = obs[i:i+chunk_size]
        batch_obs = env.preprocess_obs(batch_obs, info=preprocess_info, batch=True)
        observations.append(batch_obs.numpy())

    observations = np.concatenate(observations, axis=0)

    saved_dict = {}

    saved_dict['observations'] = observations
    for key in np_dataset:
        if key != 'observations':
            saved_dict[key] = np_dataset[key].copy()

    for key in saved_dict:
        print(key, saved_dict[key].shape)

    np.savez(save_path, **saved_dict)


def load_dataset_to_buffer(cfg, env, validation=False):
    """Load dataset to buffer for offline training."""
    # Load numpy dataset
    fp = os.path.join(cfg.data_dir, f'{get_filename(cfg, validation)}.npz')
    if not os.path.exists(fp):
        print(f"Dataset file {fp} not found, converting dataset images to {cfg.obs} observations...")
        preprocess_and_save_dataset(cfg, env, fp)
        print(f"Processed dataset saved to: {fp}")
    np_dataset = load_dataset_from_file(fp, cfg)
    
    # Get dataset attributes
    capacity = np_dataset["terminals"].size
    cfg.data_episode_length = int(np.nonzero(np_dataset["terminals"])[0][0]) + 1
    
    # Convert dataset to torch tensordict
    obs = torch.from_numpy(np_dataset["observations"])
    
    if cfg.obs in ['state', 'ec_state', 'ec_state_gen', 'rgb']:
        # Process observations in case they have not been pre-processed into a dataset
        observations = []
        chunk_size = cfg.data_episode_length + 1  # NOTE: for ec_state_gen it is important for the chunk_size to cover exactly one episode for the consistency of object IDs
        print(f"Preprocessing observations...")
        for i in trange(0, len(obs), chunk_size):
            batch_obs = obs[i:i+chunk_size]
            batch_obs = env.preprocess_obs(batch_obs, batch=True)
            observations.append(batch_obs.numpy())
        obs = np.concatenate(observations, axis=0)

    dataset = TensorDict({
            "obs": obs,
            "action": torch.from_numpy(np_dataset["actions"]),
        }, batch_size=capacity)
    
    # Add information for reward calculation to tensordict
    if cfg.reward != 'unsupervised':
        if 'manipobj' in cfg.task:
            dataset["qpos"] = torch.from_numpy(np_dataset["qpos"])
        if 'scene' in cfg.task:
            dataset["qpos"] = torch.from_numpy(np_dataset["qpos"])
            dataset["button_states"] = torch.from_numpy(np_dataset["button_states"])
        elif 'pushtetris' in cfg.task:
            dataset["image_bitmasks"] = torch.from_numpy(np_dataset["image_bitmasks"])
            
    # Create buffer for sampling
    buffer = Buffer(cfg, env, capacity)
    buffer.load(dataset)
    
    expected_episodes = capacity // (cfg.data_episode_length + 1)
    if buffer.num_eps != expected_episodes:
        print(f'WARNING: buffer has {buffer.num_eps} episodes, expected {expected_episodes} episodes for {cfg.task} task.')

    return buffer
