#!/bin/bash


#### PPP-Cube ####
python collect.py task=manipobj-v0 env_kwargs.manipobj.mode=data_collection env_kwargs.manipobj.num_cubes=3 multiview=true dataset_name=ppp-cube-noisy-v0-mv dataset_type=noisy num_collect_episodes=7500 max_episode_steps=400

#### Stack-Cube ####
python collect.py task=manipobj-v0 env_kwargs.manipobj.mode=data_collection_stack env_kwargs.manipobj.num_cubes=3 multiview=true dataset_name=stack-cube-noisy-v0-mv dataset_type=noisy p_random_action=0.02 num_collect_episodes=3000 max_episode_steps=1000

#### Scene ####
python collect.py task=visual-scene-v0 env_kwargs.scene.mode=data_collection env_kwargs.scene.num_cubes=1 multiview=false dataset_name=visual-scene-noisy-v0 dataset_type=noisy num_collect_episodes=1000 max_episode_steps=1000

#### PushTetris ####
python collect.py task=pushtetris-v0 env_kwargs.pushtetris.mode=data_collection env_kwargs.pushtetris.num_objects=3 dataset_name=push-tetris-constrained_random-v0 dataset_type=noisy num_collect_episodes=2500 max_episode_steps=400
