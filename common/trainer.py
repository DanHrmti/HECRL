from time import time
from collections import defaultdict

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from tqdm import trange

from common.utils.dataset import load_dataset_to_buffer
from common.utils.misc import add_to, flatten, visualize_frame


class Trainer:
	"""Trainer class for offline agents"""
	def __init__(self, cfg, env, agent, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.logger = logger
		self.buffer = load_dataset_to_buffer(self.cfg, self.env)

		print("Learnable parameters: {:,}".format(self.agent.total_params))
		self._start_time = time()

	def _time_metrics(self, step):
		"""Return a dictionary of currentt time metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=step,
			elapsed_time=elapsed_time,
			steps_per_second=step / elapsed_time
		)
	
	def _eval_ogbench(self):
		"""OGBench evaluation protocol"""
		metrics = {}
		overall_metrics = defaultdict(list)
		frames_list = []
		
		task_infos = self.env.unwrapped.task_infos if hasattr(self.env.unwrapped, 'task_infos') else self.env.task_infos
		
		for task_id in [1, 2, 3, 4, 5]:
			
			cur_metrics = defaultdict(list)
			cur_frames = []
			
			for i in trange(self.cfg.num_eval_episodes + self.cfg.num_video_episodes):
				should_render = i >= self.cfg.num_eval_episodes

				obs, goal_info = self.env.reset(options=dict(task_id=task_id, render_goal=should_render))
				
				goal = goal_info.get('goal')
				goal = self.buffer.normalizer.normalize(goal.unsqueeze(0))
				
				done = False
				step = 0
				frames = []
				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					obs = self.buffer.normalizer.normalize(obs.unsqueeze(0))
					action, act_info = self.agent.act(TensorDict(obs=obs, goal=goal, step=step))
					action = action.squeeze(0).cpu()

					next_obs, reward, terminated, truncated, info = self.env.step(action)
					done = terminated or truncated
					step += 1

					if should_render and (step % self.cfg.video_frame_skip == 0 or done):
						frames.append(visualize_frame(self.cfg, self.env, reward, self.buffer.obs_mean, self.buffer.obs_std, info, goal_info, act_info))

					obs = next_obs

				if i < self.cfg.num_eval_episodes:
					add_to(cur_metrics, flatten(info))
				else:
					cur_frames.append(np.array(frames))

			frames_list.extend(cur_frames)
			
			for k, v in cur_metrics.items():
				cur_metrics[k] = np.mean(v)

			task_name = task_infos[task_id - 1]['task_name']
			metric_names = ['success', 'success_frac']
			metrics.update(
				{f'{task_name}_{k}': v for k, v in cur_metrics.items() if k in metric_names}
			)
			for k, v in cur_metrics.items():
				if k in metric_names:
					overall_metrics[k].append(v)
		
		for k, v in overall_metrics.items():
				metrics[f'overall_{k}'] = np.mean(v)
		
		return metrics, frames_list  # NOTE: videos should be None if we don't want to save them
	
	def _eval(self):
		if self.cfg.ogbench or self.cfg.task == 'visual-scene-v0':
			return self._eval_ogbench()
		
		ep_rewards, ep_successes, ep_success_fracs, ep_lengths = [], [], [], []
		ep_interaction_rates = []
		frames_list = []
		if 'pushtetris' in self.cfg.task:
			ep_coverage, ep_chamfer = [], []

		for i in trange(self.cfg.num_eval_episodes):
			should_render = i < self.cfg.num_video_episodes
			frames = []
			done, ep_reward, step, interaction = False, 0, 0, 0

			obs, goal_info = self.env.reset(options=dict(render_goal=True))
			
			goal = goal_info.get('goal')
			goal = self.buffer.normalizer.normalize(goal.unsqueeze(0))
			
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				obs = self.buffer.normalizer.normalize(obs.unsqueeze(0))  # goal is normalized once in the beginning of the episode
				if self.cfg.obs == 'ec_state_gen':
					obs[..., -9:] = goal[..., -9:]
				action, act_info = self.agent.act(TensorDict(obs=obs, goal=goal, step=step))
				action = action.squeeze(0).cpu()

				next_obs, reward, terminated, truncated, info = self.env.step(action)
				done = terminated or truncated
				step += 1

				ep_reward += reward
				interaction += info.get('was_agent_object_interaction')

				if should_render and (step % self.cfg.video_frame_skip == 0 or done):
					frames.append(visualize_frame(self.cfg, self.env, reward, self.buffer.obs_mean, self.buffer.obs_std, info, goal_info, act_info))

				obs = next_obs

			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_success_fracs.append(info['success_frac'])
			ep_lengths.append(step)
			ep_interaction_rates.append(interaction / step)
			if 'pushtetris' in self.cfg.task:
				ep_coverage.append(info['final_coverage'])
				ep_chamfer.append(info['final_chamfer'])
			if i < self.cfg.num_video_episodes:
				frames_list.append(np.array(frames))

		metrics = dict(
			overall_success=np.nanmean(ep_successes),
			overall_success_frac=np.nanmean(ep_success_fracs),
			episode_reward=np.nanmean(ep_rewards),
			episode_length=np.nanmean(ep_lengths),
			episode_interaction_rate=np.nanmean(ep_interaction_rates),
		)
		if 'pushtetris' in self.cfg.task:
			metrics["episode_coverage"] = np.nanmean(ep_coverage)
			metrics["episode_chamfer"] = np.nanmean(ep_chamfer)

		return metrics, frames_list

	def train(self):
		print(f'Training agent for {self.cfg.steps} iterations...')
		best_success_rate = 0
		for i in range(self.cfg.steps):
			# sample batched data
			batch = self.buffer.sample()

			# update agent
			torch.compiler.cudagraph_mark_step_begin()
			train_metrics = self.agent.update(batch, i)

			# evaluate agent periodically
			if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
				train_metrics.update(self._time_metrics(i))
				self.logger.log(train_metrics, 'train')
				if i % self.cfg.eval_freq == 0:
					eval_metrics, eval_videos = self._eval()
					eval_metrics.update(self._time_metrics(i))
					self.logger.log(eval_metrics, 'eval', eval_videos)
					if i > 0:
						self.logger.save_agent(self.agent, self.buffer, identifier=f'latest')
						if eval_metrics["overall_success"] > best_success_rate:
							self.logger.save_agent(self.agent, self.buffer, identifier=f'best')
							best_success_rate = eval_metrics["overall_success"]
			
		self.logger.finish(self.agent, self.buffer)
		