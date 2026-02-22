from copy import deepcopy

import numpy as np
import torch

import gymnasium as gym
from gymnasium.envs.registration import register

from visual_encoders import load_pretrained_rep_model, extract_dlp_features, extract_dlp_image, extract_vqvae_features


register(
    id='manipobj-v0',
    entry_point='envs.manip_obj.manip_obj_env:ManipObjEnv',
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
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state_obs').copy())
		
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
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state_goal').copy())
		goal = self.preprocess_obs(self._obs_to_tensor(goal), info=self.preprocess_info)
		info["goal"] = goal

		# preprocess observation
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state_obs').copy())
		observation = self.preprocess_obs(self._obs_to_tensor(observation), info=self.preprocess_info)

		return observation, info    
	
	def step(self, action):
		observation, reward, terminated, truncated, info = self.env.step(action.numpy())
		
		# preprocess observation
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state_obs').copy())
		observation = self.preprocess_obs(self._obs_to_tensor(observation), info=self.preprocess_info)

		return observation, torch.tensor(reward, dtype=torch.float32), terminated, truncated, info
			
	def preprocess_obs(self, obs, info={}, batch=False):
		
		if not batch:
			obs = obs.unsqueeze(0)

		if self.obs_mode == 'state':
			pass

		elif self.obs_mode == 'rgb':
			obs = obs.permute(0, 3, 1, 2)

		elif self.obs_mode == 'ec_state':
			num_obs = obs.shape[0]
			agent_state_dim = self.env.unwrapped.agent_state_dim
			object_state_dim = self.env.unwrapped.object_state_dim

			agent_obs = obs[:, :agent_state_dim].unsqueeze(1)
			object_obs = obs[:, agent_state_dim:].reshape(num_obs, -1, object_state_dim)
			padded_object_obs = torch.cat([object_obs, torch.zeros(num_obs, object_obs.shape[1], agent_state_dim - object_state_dim)], dim=-1)
			entity_obs = torch.cat([agent_obs, padded_object_obs], dim=1)

			num_entities = entity_obs.shape[1]  # num_objects + agent
			id = torch.eye(num_entities).unsqueeze(0).expand(num_obs, -1, -1)
			
			obs = torch.cat([entity_obs, id], dim=-1)

		elif self.obs_mode == 'ec_state_gen':
			num_obs = obs.shape[0]
			agent_state_dim = self.env.unwrapped.agent_state_dim
			object_state_dim = self.env.unwrapped.object_state_dim

			agent_obs = obs[:, :agent_state_dim].unsqueeze(1)
			object_obs = obs[:, agent_state_dim:].reshape(num_obs, -1, object_state_dim)
			padded_object_obs = torch.cat([object_obs, torch.zeros(num_obs, object_obs.shape[1], agent_state_dim - object_state_dim)], dim=-1)
			entity_obs = torch.cat([agent_obs, padded_object_obs], dim=1)

			num_entities = entity_obs.shape[1]  # num_objects + agent
			num_entity_ids = 9  # 8 objects + agent
			
			all_ids = torch.eye(num_entity_ids, device=obs.device, dtype=obs.dtype)
			perm = torch.randperm(num_entity_ids - 1)[:num_entities - 1] + 1
			ids = torch.cat([all_ids[0:1], all_ids[perm]], dim=0)  # (num_entities, num_entity_ids)
			id = ids.unsqueeze(0).expand(num_obs, -1, -1)  # (num_obs, num_entities, num_entity_ids)

			id = info.get("id", id)

			obs = torch.cat([entity_obs, id], dim=-1)

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
	assert cfg.task == "manipobj-v0"
	
	ob_type = 'pixels' if cfg.obs in ['rgb', 'vqvae', 'dlp'] else 'states_clean'

	env = gym.make(cfg.task, ob_type=ob_type, multiview=cfg.multiview, max_episode_steps=cfg.max_episode_steps, **cfg.env_kwargs)
	
	cfg.env_episode_length = env._max_episode_steps

	env = TorchObsWrapper(env, cfg)

	cfg.num_objects = env.unwrapped.num_objects
	cfg.num_cubes = env.unwrapped.num_cubes

	return env
