import numpy as np
import torch

import gymnasium as gym
from gymnasium.envs.registration import register

from visual_encoders import load_pretrained_rep_model, extract_dlp_features, extract_dlp_image, extract_vqvae_features


register(
    id="pushtetris-v0",
    entry_point="envs.push_tetris.push_tetris_env:PushTetrisEnv",
    max_episode_steps=400,
)


class TorchObsWrapper(gym.Wrapper):
    
	def __init__(self, env, cfg):
		super().__init__(env)
		
		self.cfg = cfg
		self.obs_mode = cfg.obs
		self.preprocess_info = {
        	"rep_model": load_pretrained_rep_model(cfg),
			"device": cfg.rep_model_device,
		}

		obs, info = self.env.reset()
		
		obs = self.preprocess_obs(self._obs_to_tensor(obs), info=self.preprocess_info)
		obs = obs.numpy()
		
		if self.obs_mode == 'rgb':
			self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs.shape, dtype=obs.dtype)
		else:
			self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=obs.dtype)

	def rand_act(self):
		return torch.from_numpy(self.action_space.sample()).float()
	
	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs
	
	def reset(self, *, seed=None, options=None):
		observation, info = self.env.reset(seed=seed, options=options)
		
		# preprocess goal
		goal = info.get('goal')
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state_goal'))
		goal = self.preprocess_obs(self._obs_to_tensor(goal), info=self.preprocess_info)
		info["goal"] = goal

		# preprocess observation
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state'))
		observation = self.preprocess_obs(self._obs_to_tensor(observation), info=self.preprocess_info)

		return observation, info    
	
	def step(self, action):
		observation, reward, terminated, truncated, info = self.env.step(action.numpy())
		
		# preprocess observation
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state'))
		observation = self.preprocess_obs(self._obs_to_tensor(observation), info=self.preprocess_info)

		return observation, torch.tensor(reward, dtype=torch.float32), terminated, truncated, info
			
	def preprocess_obs(self, obs, batch=False, info={}):
	
		if not batch:
			obs = obs.unsqueeze(0)

		if self.obs_mode == 'state':
			pass

		elif self.obs_mode == 'rgb':
			obs = obs.permute(0, 3, 1, 2)

		elif self.obs_mode == 'ec_state':
			num_obs = obs.shape[0]
			obs = obs.reshape(obs.shape[0], self.env.unwrapped.num_objects + 1, -1)

		elif self.obs_mode == 'dlp':
			obs = obs.to(torch.device(info.get("device", 'cuda:0')))
			obs = extract_dlp_features(obs, info["rep_model"])
			obs = obs.cpu()
		
		elif self.obs_mode == 'vqvae':
			obs = obs.to(torch.device(info.get("device", 'cuda:0')))
			obs = extract_vqvae_features(obs, info["rep_model"])
			obs = obs.cpu()
		
		else:
			raise NotImplementedError

		if not batch:
			obs = obs.squeeze(0)

		return obs
	
	def render(self, info={}):
		frame = self.env.unwrapped.render()
		
		if self.obs_mode == 'dlp':
			if frame.ndim == 4:  # multiview
				n_views = frame.shape[0]
				frame = np.stack([
					extract_dlp_image(np.transpose(frame[i], (2, 0, 1)), info["obs_mean"], info["obs_std"], self.preprocess_info["rep_model"], device=self.preprocess_info["device"])
					for i in range(n_views)], axis=0)
			else:
				frame = extract_dlp_image(np.transpose(frame, (2, 0, 1)), info["obs_mean"], info["obs_std"], self.preprocess_info["rep_model"], device=self.preprocess_info["device"])

		return frame
		

def make_env(cfg):
	assert cfg.task == "pushtetris-v0"
	
	ob_type = 'pixels' if cfg.obs in ['rgb', 'vqvae', 'dlp'] else 'states'
	
	env = gym.make(cfg.task, ob_type=ob_type, max_episode_steps=cfg.max_episode_steps, **cfg.env_kwargs)
	
	cfg.env_episode_length = env._max_episode_steps

	env = TorchObsWrapper(env, cfg)

	cfg.num_objects = env.unwrapped.num_objects

	return env
