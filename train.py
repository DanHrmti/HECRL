import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
# os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

# Torch compile: allow more recompilations before falling back to eager (default 8).
# Raising this can improve performance if your workload triggers many recompiles
# (e.g. tensor shape/rank changes from tensordict). After the limit, dynamo falls
# back to eager for that function, so a higher limit keeps compiled code longer.
try:
	torch.compiler.config.recompile_limit = 32
except AttributeError:
	torch._dynamo.config.recompile_limit = 32

import hydra
from termcolor import colored

from envs import make_env
from agents import make_agent

from common.trainer import Trainer

from common.utils.logger import Logger
from common.utils.parser import parse_cfg
from common.utils.misc import set_seed

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training an offline agent.

	Most relevant args:
		`task`: environment/task name
		`agent`: type of agent to train
		`steps`: number of training steps (default: 10M)
		`seed`: random seed (if 'random' will choose randomly from [0, 50_000], default: 'random')

	See config.yaml for a full list of args.
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	trainer = Trainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=make_agent(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
	