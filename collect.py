import os
os.environ['MUJOCO_GL'] = 'egl'

import hydra
from termcolor import colored

from common.utils.parser import parse_cfg
from common.utils.misc import set_seed

from datasets.collect_manipobj import collect as collect_manipobj
from datasets.collect_pushtetris import collect as collect_pushtetris
from datasets.collect_ogbench import collect as collect_ogbench


@hydra.main(config_name='config', config_path='.')
def collect(cfg: dict):
    """
    Script for collecting an offline dataset.

    Most relevant args:
        `task`: task/env name
        `dataset_type`: type of dataset to collect, applies for OGBench-based environments
            'play': Use a non-Markovian oracle (PlanOracle) that follows a pre-computed plan
            'noisy': Use a Markovian, closed-loop oracle (MarkovOracle) with Gaussian action noise
        `num_collect_episodes`: number of episodes to collect
        `seed`: random seed (if 'random' will choose randomly from [0, 50_000], default: 'random')

    See config.yaml for a full list of args.
    """
    assert cfg.num_collect_episodes > 0, 'Must collect at least 1 episode.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored(f'Environment: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Dataset Type: {cfg.dataset_type}', 'blue', attrs=['bold']))
    print(colored(f'Dataset Name: {cfg.dataset_name}', 'blue', attrs=['bold']))
    print(colored(f'Number of Episodes: {cfg.num_collect_episodes}', 'blue', attrs=['bold']))
    print(colored(f'Episode Length: {cfg.max_episode_steps}', 'blue', attrs=['bold']))

    if cfg.task == 'manipobj-v0':
        collect_manipobj(cfg)
    elif cfg.task == 'pushtetris-v0':
        collect_pushtetris(cfg)
    elif cfg.task == 'visual-scene-v0':
        collect_ogbench(cfg)
    else:
        raise ValueError(f'Unknown task: {cfg.task}')
    

if __name__ == '__main__':
    collect()
    