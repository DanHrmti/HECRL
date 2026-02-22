from collections import defaultdict
import os

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from tqdm import trange

from moviepy import ImageSequenceClip

import ogbench.manipspace  # noqa
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle


register(
    id='visual-scene-v0',
    entry_point="envs.scene.scene_env:SceneEnv",
    max_episode_steps=1000,
    kwargs=dict(
        ob_type='pixels',
		width=128,
		height=128,
		visualize_info=False,
		terminate_at_goal=False,
    ),
)


def collect(cfg):
    # Initialize environment
    env = gym.make(cfg.task, ob_type='pixels', multiview=cfg.multiview, max_episode_steps=cfg.max_episode_steps, **cfg.env_kwargs)
    num_cubes = env.unwrapped._num_cubes

    # Initialize oracles
    oracle_type = 'plan' if cfg.dataset_type == 'play' else 'markov'
    has_button_states = hasattr(env.unwrapped, '_cur_button_states')
    
    if 'cube' in cfg.task:
        if oracle_type == 'markov':
            agents = {
                'cube': CubeMarkovOracle(env=env, min_norm=cfg.min_action_norm),
            }
        else:
            agents = {
                'cube': CubePlanOracle(env=env, noise=cfg.action_noise, noise_smoothing=cfg.action_noise_smoothing),
            }
    elif 'scene' in cfg.task:
        if oracle_type == 'markov':
            agents = {
                'cube': CubeMarkovOracle(env=env, min_norm=cfg.min_action_norm, max_step=100),
                'button': ButtonMarkovOracle(env=env, min_norm=cfg.min_action_norm),
                'drawer': DrawerMarkovOracle(env=env, min_norm=cfg.min_action_norm),
                'window': WindowMarkovOracle(env=env, min_norm=cfg.min_action_norm),
            }
        else:
            agents = {
                'cube': CubePlanOracle(env=env, noise=cfg.action_noise, noise_smoothing=cfg.action_noise_smoothing),
                'button': ButtonPlanOracle(env=env, noise=cfg.action_noise, noise_smoothing=cfg.action_noise_smoothing),
                'drawer': DrawerPlanOracle(env=env, noise=cfg.action_noise, noise_smoothing=cfg.action_noise_smoothing),
                'window': WindowPlanOracle(env=env, noise=cfg.action_noise, noise_smoothing=cfg.action_noise_smoothing),
            }
    elif 'puzzle' in cfg.task:
        if oracle_type == 'markov':
            agents = {
                'button': ButtonMarkovOracle(env=env, min_norm=cfg.min_action_norm, gripper_always_closed=True),
            }
        else:
            agents = {
                'button': ButtonPlanOracle(
                    env=env,
                    noise=cfg.action_noise,
                    noise_smoothing=cfg.action_noise_smoothing,
                    gripper_always_closed=True,
                ),
            }

    # Collect data
    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = cfg.num_collect_episodes
    num_val_episodes = cfg.num_collect_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        # Have an additional while loop to handle rare cases with undesirable states (for the Scene environment).
        while True:

            make_video = (ep_idx == 0)
            if make_video:
                frames = []

            ob, info = env.reset()

            # Set the cube stacking probability for this episode
            if 'scene' in cfg.task:
                p_stack = 0.0
            elif num_cubes == 1:
                p_stack = 0.0
            elif num_cubes == 2:
                p_stack = np.random.uniform(0.0, 0.25)
            elif num_cubes == 3:
                p_stack = np.random.uniform(0.05, 0.35)
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
            ep_qpos = []

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
                if has_button_states:
                    dataset['button_states'].append(info['prev_button_states'])
                ep_qpos.append(info['prev_qpos'])

                ob = next_ob
                step += 1

                if make_video:
                    frame = env.unwrapped.get_pixel_observation()
                    if len(frame.shape) == 4:
                        frame = np.concatenate([frame[i] for i in range(len(frame))], axis=1)
                    frames.append(frame)
                
            if 'scene' in cfg.task:
                # Perform health check. We want to ensure that the cube is always visible unless it's in the drawer.
                # Otherwise, the test-time goal images may become ambiguous.
                is_healthy = True
                ep_qpos = np.array(ep_qpos)
                block_xyzs = ep_qpos[:, 14:17]
                if (block_xyzs[:, 1] >= 0.29).any():
                    is_healthy = False  # Block goes too far right.
                if ((block_xyzs[:, 1] <= -0.3) & ((block_xyzs[:, 2] < 0.06) | (block_xyzs[:, 2] > 0.08))).any():
                    is_healthy = False  # Block goes too far left, without being in the drawer.

                if is_healthy:
                    break
                else:
                    # Remove the last episode and retry.
                    print('Unhealthy episode, retrying...', flush=True)
                    for k in dataset.keys():
                        dataset[k] = dataset[k][:-step]
            else:
                break
        
        if make_video:
            frames_array = np.array(frames)
            clip = ImageSequenceClip([frames_array[i] for i in range(len(frames_array))], fps=15)
            clip.write_videofile(f'{cfg.work_dir}/output/collect_video_{cfg.task}.mp4')
            # exit()  # NOTE: uncomment for debug

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

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
        elif k == 'button_states':
            dtype = np.int64
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)
    
    print(f"Datasets saved to:\n{train_path}\n{val_path}")
