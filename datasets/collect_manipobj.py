from collections import defaultdict
import os

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from tqdm import trange

from moviepy import ImageSequenceClip

import ogbench.manipspace  # noqa

from datasets.collection_policies.pickplace_cube import PickPlaceCubeMarkovOracle
from datasets.collection_policies.push_cube import PushCubeMarkovOracle


register(
    id='manipobj-v0',
    entry_point='envs.manip_obj.manip_obj_env:ManipObjEnv',
)


def collect(cfg):
    # Initialize environment
    env = gym.make(cfg.task, ob_type='pixels', multiview=True, max_episode_steps=cfg.max_episode_steps, **cfg.env_kwargs)
    num_cubes = env.unwrapped.num_cubes

    # Initialize oracles
    oracle_type = 'plan' if cfg.dataset_type == 'play' else 'markov'
    
    agents = {
            'pickplace_cube': PickPlaceCubeMarkovOracle(max_step=200, env=env, min_norm=cfg.min_action_norm),
            'push_cube': PushCubeMarkovOracle(max_step=100, env=env, min_norm=cfg.min_action_norm),
        }

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

        # Set the cube stacking probability for this episode
        if env.unwrapped._mode == "data_collection":
            if num_cubes > 1:
                p_stack = 0.5
            else:
                p_stack = 0.0
        else:  # env.unwrapped._mode == "data_collection_stack"
            if num_cubes == 1:
                p_stack = 0.0
            elif num_cubes == 2:
                p_stack = np.random.uniform(0.0, 0.25)
            elif num_cubes == 3:
                # p_stack = np.random.uniform(0.05, 0.35)
                p_stack = 0.9
            elif num_cubes == 4:
                p_stack = np.random.uniform(0.1, 0.5)
            else:
                p_stack = 0.5

        if oracle_type == 'markov':
            # Set the action noise level for this episode
            xi = np.random.uniform(0, cfg.action_noise)

        agent = agents[info['privileged/target_task']]
        agent.reset(ob, info)

        done = False
        step = 0

        while not done:
            if np.random.rand() < cfg.p_random_action:
                # Sample a random action
                action = env.action_space.sample()
            else:
                # Get an action from the oracle
                action = agent.select_action(ob, info)
                action = np.array(action)
                if oracle_type == 'markov':
                    # Add Gaussian noise to the action
                    action = action + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 5], action.shape)  # NOTE: decreaseed noise multiplier from 10 to 5 in gripper to decrease dropping of picked cubes
            action = np.clip(action, -1, 1)
            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if agent.done:
                # Set a new task when the current task is done
                agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                agent = agents[agent_info['privileged/target_task']]
                agent.reset(agent_ob, agent_info)

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['state_observations'].append(info['prev_state_clean_observation'])

            ob = next_ob
            step += 1

            if make_video:
                frame = env.unwrapped.get_pixel_observation()
                if len(frame.shape) == 4:
                    frame = np.concatenate([frame[i] for i in range(len(frame))], axis=1)
                frames.append(frame)

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

        if make_video:
            task_name = "stackcube" if env.unwrapped._mode == "data_collection_stack" else "pppcube"
            frames_array = np.array(frames)
            clip = ImageSequenceClip([frames_array[i] for i in range(len(frames_array))], fps=15)
            clip.write_videofile(f'{cfg.work_dir}/output/collect_video_{task_name}.mp4')
            # exit()  # NOTE: uncomment for debug

    print('Total steps:', total_steps)

    train_path = os.path.join(cfg.data_dir, f'{cfg.dataset_name}.npz')
    val_path = train_path.replace('.npz', '-val.npz')

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
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
