import numpy as np
import torch

import gymnasium as gym
from gymnasium.envs.registration import register

from visual_encoders import load_pretrained_rep_model, extract_dlp_features, extract_dlp_image, extract_vqvae_features

from ogbench.manipspace.envs.scene_env import SceneEnv


register(
    id='visual-scene-v0',
    entry_point="envs.scene.scene_env:SceneEnv",
    max_episode_steps=750,
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
		return torch.from_numpy(self.action_space.sample().astype(np.float32))
	
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
		goal = self.preprocess_obs(self._obs_to_tensor(goal), info=self.preprocess_info)
		info["goal"] = goal

		# preprocess observation
		observation = self.preprocess_obs(self._obs_to_tensor(observation), info=self.preprocess_info)

		return observation, info    
	
	def step(self, action):
		observation, reward, terminated, truncated, info = self.env.step(action.numpy())
		
		# preprocess observation
		observation = self.preprocess_obs(self._obs_to_tensor(observation), info=self.preprocess_info)

		return observation, torch.tensor(reward, dtype=torch.float32), terminated, truncated, info
			
	def preprocess_obs(self, obs, info={}, batch=False):
		
		if not batch:
			obs = obs.unsqueeze(0)

		if self.obs_mode == 'rgb':
			obs = obs.permute(0, 3, 1, 2)

		elif self.obs_mode == 'dlp':
			obs = obs.to(torch.device(info.get("device", 'cuda:0')))
			if obs.ndim == 5:  # multiview
				n_views = obs.shape[1]
				obs = torch.cat([extract_dlp_features(obs[:, i], info["rep_model"]) for i in range(n_views)], dim=1)
			else:
				obs = extract_dlp_features(obs, info["rep_model"])
			obs = obs.cpu()
		
		elif self.obs_mode == 'vqvae':
			obs = obs.to(torch.device(info.get("device", 'cuda:0')))
			if obs.ndim == 5:  # multiview
				n_views = obs.shape[1]
				obs = torch.cat([extract_vqvae_features(obs[:, i], info["rep_model"]) for i in range(n_views)], dim=1)
			else:
				obs = extract_vqvae_features(obs, info["rep_model"])
			obs = obs.cpu()
		
		else:
			raise NotImplementedError

		if not batch:
			obs = obs.squeeze(0)

		return obs
	
	def render(self, info={}):
		frame = self.env.unwrapped.get_pixel_observation()
		
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
	assert cfg.task == "visual-scene-v0"
	
	assert cfg.obs in ['rgb', 'vqvae', 'dlp'], f"{cfg.task} only supports image-based observations, got {cfg.obs}"
	ob_type = 'pixels'
	
	env = gym.make(cfg.task, ob_type=ob_type, multiview=cfg.multiview, max_episode_steps=cfg.max_episode_steps, **cfg.env_kwargs)
	
	cfg.env_episode_length = env._max_episode_steps

	env = TorchObsWrapper(env, cfg)

	return env
