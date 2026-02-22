import gymnasium as gym

import ogbench
from envs.manip_obj import make_env as make_manip_obj_env
from envs.push_tetris import make_env as make_push_tetris_env
from envs.scene import make_env as make_scene_env


def make_ogbench_env(cfg):
	env = ogbench.make_env_and_datasets(cfg.task, env_only=True)
	if 'visual' in cfg.task:
		assert cfg.obs in ['rgb', 'vqvae', 'dlp']
	return env


def make_env(cfg):
	"""
	Make an environment for evaluation of offline agent.
	"""
	env = None
	for fn in [make_manip_obj_env, make_scene_env, make_push_tetris_env, make_ogbench_env]:
		try:
			env = fn(cfg)
			break
		except:
			pass
	if env is None:
		raise ValueError(f'Failed to make environment "{cfg.task}"')
	
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = env.observation_space.shape
	
	cfg.action_dim = env.action_space.shape[0]
	cfg.action_high = env.action_space.high[0].item()
	cfg.action_low = env.action_space.low[0].item()

	return env
