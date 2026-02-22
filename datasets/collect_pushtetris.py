from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from tqdm import trange

from moviepy import ImageSequenceClip


register(
    id="pushtetris-v0",
    entry_point="envs.push_tetris.push_tetris_env:PushTetrisEnv",
    max_episode_steps=400,
)


def collect(cfg):
    # Initialize environment
    env = gym.make(
        cfg.task,
        max_episode_steps=cfg.max_episode_steps,

        object_list=["O", "I", "S", "Z", "L", "J", "T"],
        num_objects=3,

        mode='data_collection',
        ob_type='pixels',

        visualize_target=False,
        render_size=128,
    )

    # Collect data
    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = cfg.num_collect_episodes
    num_val_episodes = cfg.num_collect_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):

        make_video = (ep_idx == 0)
        if make_video:
            frames = []

        ob, info = env.reset()

        done = False
        step = 0

        while not done:
            action = env.unwrapped.get_data_collection_action()
            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['state_observations'].append(info['prev_state'])
            dataset['image_bitmasks'].append(info['prev_image_bitmasks'])

            ob = next_ob
            step += 1

            if make_video:
                frame = env.unwrapped.render(mode="rgb_array")
                frames.append(frame)

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

        if make_video:
            frames_array = np.array(frames)
            clip = ImageSequenceClip([frames_array[i] for i in range(len(frames_array))], fps=15)
            clip.write_videofile(f'{cfg.work_dir}/output/collect_video_pushtetris.mp4')
            # exit()  # NOTE: uncomment for debug

    print('Total steps:', total_steps)

    train_path = os.path.join(cfg.data_dir, f'{cfg.dataset_name}.npz')
    val_path = train_path.replace('.npz', '-val.npz')

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if (k == 'observations' and v[0].dtype == np.uint8) or k == 'image_bitmasks':
            dtype = np.uint8
        elif k == 'terminals':
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)
    
    print(f"Datasets saved to:\n{train_path}\n{val_path}")
