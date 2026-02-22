import dataclasses
import os
import datetime
import re

import numpy as np
from termcolor import colored

from PIL import Image, ImageEnhance


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("overall_success", "S", "float"),
	("overall_success_frac", "SF", "float"),
	("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	kvs = [
		("environment", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observation", cfg.obs),
		("action dim", cfg.action_dim),
		("agent", cfg.agent),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, cfg.agent, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


def reshape_video(v, n_cols=5):
	"""Helper function to reshape videos."""
	if v.ndim == 4:
		v = v[None,]

	_, t, h, w, c = v.shape

	if n_cols is None:
		# Set n_cols to the square root of the number of videos.
		n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
	if v.shape[0] % n_cols != 0:
		len_addition = n_cols - v.shape[0] % n_cols
		v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
	n_rows = v.shape[0] // n_cols

	v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
	v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
	v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

	return v


def frames_list_to_array(frames_list=None, n_cols=5):
	"""

	It takes a list of videos and reshapes them into a single video with the specified number of columns.

	Args:
		frames_list: List of videos. Each video should be a numpy array of shape (t, h, w, c).
		n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
	"""
	# Pad videos to the same length.
	max_length = max([len(frames) for frames in frames_list])
	for i, frames in enumerate(frames_list):
		assert frames.dtype == np.uint8

		# Decrease brightness of the padded frames.
		final_frame = frames[-1]
		final_image = Image.fromarray(final_frame)
		enhancer = ImageEnhance.Brightness(final_image)
		final_image = enhancer.enhance(0.5)
		final_frame = np.array(final_image)

		pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(frames), axis=0)
		frames_list[i] = np.concatenate([frames, pad], axis=0)

		# Add borders.
		frames_list[i] = np.pad(frames_list[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
	frames_array = np.array(frames_list)  # (n, t, h, w, c)

	frames_array = reshape_video(frames_array, n_cols)  # (t, c, nr * h, nc * w)

	return frames_array
		

class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self._log_dir = make_dir(cfg.log_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_agent = cfg.save_agent
		self._save_video = cfg.save_video
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = False
			self._wandb = None
			self._video = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb

		wandb.init(
			project=self.project,
			entity=self.entity,
			name=f"{cfg.exp_name}{cfg.wandb_name_suffix}",
			group=self._group,
			tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
			dir=self._log_dir,
			config=dataclasses.asdict(cfg),
		)
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb

	def save_agent(self, agent, buffer, identifier='latest'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp, buffer.obs_mean, buffer.obs_std)
			# if self._wandb:
			# 	artifact = self._wandb.Artifact(
			# 		self._group + '-' + str(self._seed) + '-' + str(identifier),
			# 		type='model',
			# 	)
			# 	artifact.add_file(fp)
			# 	self._wandb.log_artifact(artifact)

	def finish(self, agent, buffer):
		try:
			self.save_agent(agent, buffer)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def _log_video(self, frames_list, step):
		frames_array = frames_list_to_array(frames_list, n_cols=len(frames_list))
		video = self._wandb.Video(frames_array, fps=15, format='mp4')
		self._wandb.log({'eval_video': video}, step=step)
	
	def log(self, d, category="train", frames_list=None):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			step = d["step"]
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			self._wandb.log(_d, step)
			
			if category == "eval" and self._save_video:
				self._log_video(frames_list, step)

		self._print(d, category)
	