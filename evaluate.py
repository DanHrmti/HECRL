import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from tensordict.tensordict import TensorDict

import hydra
from tqdm import trange
from termcolor import colored
from moviepy import ImageSequenceClip

from envs import make_env
from agents import make_agent

from common.utils.parser import parse_cfg
from common.utils.misc import set_seed, add_to, flatten, visualize_frame, calc_num_entity_mods
from common.utils.logger import frames_list_to_array

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def eval_ogbench(cfg, env, agent):
    obs_mean, obs_std = agent.obs_mean, agent.obs_std

    metrics = {}
    overall_metrics = defaultdict(list)
    frames_list = []

    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos

    for task_id in cfg.eval_task_ids:
        
        cur_metrics = defaultdict(list)
        cur_frames = []
        
        for i in trange(cfg.num_eval_episodes + cfg.num_video_episodes):
            should_render = i >= cfg.num_eval_episodes

            obs, goal_info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
            
            goal = goal_info.get('goal')
            goal = (goal.unsqueeze(0) - obs_mean) / obs_std

            done = False
            step = 0
            ep_reward = 0
            frames = []

            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                obs = (obs.unsqueeze(0) - obs_mean) / obs_std
                action, act_info = agent.act(TensorDict(obs=obs, goal=goal, step=step))
                action = action.squeeze(0).cpu()

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward

                if should_render and (step % cfg.video_frame_skip == 0 or done):
                    frames.append(visualize_frame(cfg, env, reward, obs_mean, obs_std, info, goal_info, act_info))
                
                obs = next_obs
                step += 1

            if i < cfg.num_eval_episodes:
                info['episode_reward'] = ep_reward
                add_to(cur_metrics, flatten(info))
            else:
                cur_frames.append(np.array(frames))

        frames_list.extend(cur_frames)
        
        for k, v in cur_metrics.items():
            cur_metrics[k] = np.mean(v)

        task_name = task_infos[task_id - 1]['task_name']
        metric_names = ['success', 'success_frac', 'episode_reward']
        metrics.update(
            {f'{task_name}_{k}': v for k, v in cur_metrics.items() if k in metric_names}
        )
        for k, v in cur_metrics.items():
            if k in metric_names:
                overall_metrics[k].append(v)

    for k, v in overall_metrics.items():
            metrics[f'overall_{k}'] = np.mean(v)

    return metrics, frames_list


def eval(cfg, env, agent):
    obs_mean, obs_std = agent.obs_mean, agent.obs_std
    
    ep_rewards, ep_successes, ep_success_fracs, ep_lengths = [], [], [], []
    ep_interaction_rates = []
    frames_list = []
    if 'pushtetris' in cfg.task:
        ep_coverage, ep_chamfer = [], []
    if cfg.subgoal_factor_eval:
        ep_entity_mods = []

    for i in trange(cfg.num_eval_episodes):
        should_render = i < cfg.num_video_episodes
        frames = []
        done, ep_reward, step = False, 0, 0
        interaction = 0
        if cfg.subgoal_factor_eval:
            num_entity_mods = []

        obs, goal_info = env.reset(options=dict(render_goal=True))
        
        goal = goal_info.get('goal')
        goal = (goal.unsqueeze(0) - obs_mean) / obs_std
        
        while not done:
            torch.compiler.cudagraph_mark_step_begin()
            obs = (obs.unsqueeze(0) - obs_mean) / obs_std  # goal is normalized once in the beginning of the episode
            if cfg.obs == 'ec_state_gen':  # NOTE: make sure the entity ids are consistent between goal and obs
                    obs[..., -9:] = goal[..., -9:]
            action, act_info = agent.act(TensorDict(obs=obs, goal=goal, step=step))
            action = action.squeeze(0).cpu()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            interaction += info.get('was_agent_object_interaction')

            if should_render and (step % cfg.video_frame_skip == 0 or done):
                frames.append(visualize_frame(cfg, env, reward, obs_mean, obs_std, info, goal_info, act_info))
            
            if cfg.subgoal_factor_eval:
                num_entity_mods.append(calc_num_entity_mods(cfg, env, obs, act_info["subgoal"], obs_mean, obs_std))
                done = True

            obs = next_obs
            step += 1

        ep_rewards.append(ep_reward)
        ep_successes.append(info['success'])
        ep_success_fracs.append(info['success_frac'])
        ep_lengths.append(step)
        ep_interaction_rates.append(interaction / step)
        if 'pushtetris' in cfg.task:
            ep_coverage.append(info['final_coverage'])
            ep_chamfer.append(info['final_chamfer'])
        if cfg.subgoal_factor_eval:
            ep_entity_mods.append(np.mean(num_entity_mods))
        if i < cfg.num_video_episodes:
            frames_list.append(np.array(frames))

    metrics = dict(
        overall_success=np.nanmean(ep_successes),
        overall_success_frac=np.nanmean(ep_success_fracs),
        episode_reward=np.nanmean(ep_rewards),
        # episode_length=np.nanmean(ep_lengths),
        episode_interaction_rate=np.nanmean(ep_interaction_rates),
        episode_entity_mods=np.nanmean(ep_interaction_rates),
    )
    if 'pushtetris' in cfg.task:
        metrics["episode_coverage"] = np.nanmean(ep_coverage)
        metrics["episode_chamfer"] = np.nanmean(ep_chamfer)
    if cfg.subgoal_factor_eval:
        metrics["episode_entity_mods"] = np.nanmean(ep_entity_mods)
    
    return metrics, frames_list


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
    """
    Script for evaluating an offline agent.

    Most relevant args:
        `task`: environment/task name
        `agent`: type of agent to evaluate
        `num_eval_episodes`: number of episodes to evaluate on (per task in case of OGBench tasks)
        `seed`: random seed (if 'random' will choose randomly from [0, 50_000], default: 'random')

    See config.yaml for a full list of args.
    """
    assert torch.cuda.is_available()
    assert cfg.num_eval_episodes > 0, 'Must evaluate at least 1 episode.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored(f'Environment: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Observation: {cfg.obs}', 'blue', attrs=['bold']))
    print(colored(f'Agent: {cfg.agent}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.eval_checkpoint}', 'blue', attrs=['bold']))
    print(colored(f'Evaluation Variant: {cfg.eval_variant_name}', 'blue', attrs=['bold']))

    # Make environment
    env = make_env(cfg)

    # Load agent
    agent = make_agent(cfg)
    assert os.path.exists(cfg.eval_checkpoint), f'Checkpoint {cfg.eval_checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.eval_checkpoint)

    # Evaluate
    eval_func = eval_ogbench if (cfg.ogbench or cfg.task == 'visual-scene-v0') else eval
    metrics, frames_list = eval_func(cfg, env, agent)
    
    # Logging
    os.makedirs(cfg.eval_log_dir, exist_ok=True)
    
    with open(os.path.join(cfg.eval_log_dir, 'metrics.txt'), "w") as file:
        file.write(f"Evaluation Metrics\n\n")
        file.write(f"checkpoint: {cfg.eval_checkpoint}\n")
        file.write(f"{cfg.eval_variant_name}\n\n")
        
        for key, value in metrics.items():
            data = f"{key}: {value}\n"
            file.write(data)
            print(data)

    print(f'Finished evaluating variant: {cfg.eval_variant_name}\n')
    
    if cfg.save_video:
        frames_array = frames_list_to_array(frames_list, n_cols=len(frames_list))
        frames_array = np.transpose(frames_array, (0, 2, 3, 1))
        clip = ImageSequenceClip([frames_array[i] for i in range(len(frames_array))], fps=15)
        clip.write_videofile(os.path.join(cfg.eval_log_dir, f'eval_video.mp4'))


if __name__ == '__main__':
    evaluate()
    