import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import ogbench

data_dir = 'path/for/data/to/be/saved'
path_to_npz = 'path/to/dataset/npz/file.npz'
path_to_npz_val = path_to_npz.replace('.npz', '-val.npz')
data_div = 3  # factor to divide the dataset size for training, representation training may require less data than RL training, set to 1 for full dataset

# create dirs
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

for fp in [path_to_npz, path_to_npz_val]:
    
    print(f"Processing data from: {fp}")
    # np_dataset = ogbench.load_dataset(fp, ob_dtype=np.uint8, compact_dataset=True)
    np_dataset = np.load(fp)

    images = np_dataset['observations'][:np_dataset['observations'].shape[0] // data_div]
    actions = np_dataset['actions'][:np_dataset['actions'].shape[0] // data_div]
    print(f"Dataset loaded")

    h, w, c = images.shape[-3], images.shape[-2], images.shape[-1]
    n_views = images.shape[1] if len(images.shape) == 5 else 1

    _, action_dim = actions.shape
    data_episode_length = int(np.nonzero(np_dataset["terminals"])[0][0]) + 1
    print(f"Data Episode Length: {data_episode_length}")
    print(f"Number of Views: {n_views}")

    images = images.reshape(-1, data_episode_length * n_views, h, w, c)
    actions = actions.reshape(-1, data_episode_length, action_dim)
   
    total_episodes = images.shape[0]
    print(f'Number of episodes: {total_episodes}')

    save_dir = valid_dir if 'val' in fp else train_dir
    print(f"Saving processed data in: {save_dir}")

    for ep in tqdm(range(images.shape[0])):
        ep_dir = os.path.join(save_dir, str(ep))
        os.makedirs(ep_dir, exist_ok=True)
        # save actions
        np.savez(os.path.join(ep_dir, f'actions.npz'), actions=actions[ep, :-1])
        # save images
        for i in range(images.shape[1]):
            im = images[ep][i]
            Image.fromarray(im).save(os.path.join(ep_dir, f'{i}.png'))
            