import dataclasses
import re
from pathlib import Path
from typing import Any
import random

import hydra
from omegaconf import OmegaConf


def cfg_to_dataclass(cfg, frozen=False):
	"""
	Converts an OmegaConf config to a dataclass object.
	This prevents graph breaks when used with torch.compile.
	"""
	cfg_dict = OmegaConf.to_container(cfg)
	fields = []
	for key, value in cfg_dict.items():
		fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
	dataclass_name = "Config"
	dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)
	def get(self, val, default=None):
		return getattr(self, val, default)
	dataclass.get = get
	return dataclass()


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parses a Hydra config. Mostly for convenience.
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Seed
	if cfg.seed == 'random':
		cfg.seed = random.randint(0, 50_000)

	# Environment
	assert cfg.task in ['manipobj-v0', 'pushtetris-v0', 'visual-scene-v0']
	cfg.env_kwargs = cfg.env_kwargs["manipobj"] if "manipobj" in cfg.task else \
					 cfg.env_kwargs["scene"] if "scene" in cfg.task else \
					 cfg.env_kwargs["pushtetris"] if "pushtetris" in cfg.task else \
					 {}
	
	assert cfg.reward in ['unsupervised', 'env']
	
	assert cfg.obs in ['state', 'ec_state', 'ec_state_gen', 'rgb', 'dlp', 'vqvae']
	cfg.multiview = cfg.multiview and cfg.obs in ['dlp', 'vqvae'] and cfg.task in ['manipobj-v0']  # NOTE: only manipobj-v0 currently supports multiview observations
	
	# Agent
	assert cfg.agent in ['iql', 'hiql', 'sgiql']
	agent_cfg = {'iql': cfg.iql, 'hiql': cfg.hiql, 'sgiql': cfg.sgiql}.get(cfg.agent, {})
	cfg.alpha = agent_cfg.get("alpha")
	cfg.beta = agent_cfg.get("beta")
	cfg.subgoal_k = agent_cfg.get("subgoal_k")
	cfg.subgoal_steps = agent_cfg.get("subgoal_steps")
	cfg.chamfer_metric = agent_cfg.get("chamfer_metric")
	cfg.chamfer_target_weight = agent_cfg.get("chamfer_target_weight")
	
	# Paths
	cfg.work_dir = Path(hydra.utils.get_original_cwd())
	cfg.data_dir = Path(hydra.utils.get_original_cwd()) / f'{cfg.data_dir}'
	cfg.log_dir = Path(hydra.utils.get_original_cwd()) / 'output' / 'train' / f'{cfg.agent}_{cfg.task}_{str(cfg.seed)}'
	cfg.rep_model_checkpoint = Path(hydra.utils.get_original_cwd()) / f'{cfg.rep_model_checkpoint}'
	cfg.checkpoint = Path(hydra.utils.get_original_cwd()) / f'{cfg.checkpoint}'
	cfg.sgiql["goal_pi_chkpt"] = Path(hydra.utils.get_original_cwd()) / f'{cfg.sgiql["goal_pi_chkpt"]}'
	cfg.eval_log_dir = Path(hydra.utils.get_original_cwd()) / 'output' / 'eval' / f'{cfg.agent}_{cfg.task}_{str(cfg.seed)}'
	cfg.eval_checkpoint = Path(hydra.utils.get_original_cwd()) / f'{cfg.eval_checkpoint}'
	
	# Convenience
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.exp_name = f"{cfg.agent}_{cfg.task}"

	# Evaluation
	if cfg.subgoal_factor_eval:
		assert cfg.task in ['manipobj-v0'] and \
			cfg.agent in ['sgiql', 'hiql'] and \
			cfg.obs in ['state', 'ec_state', 'ec_state_gen'], \
				'Only state-based SGIQL and HIQL on cube manipulation tasks can be evaluated for subgoal factoredness.'

	return cfg_to_dataclass(cfg)
