#!/bin/bash


#### PPP-Cube (State) ####

## EC-SGIQL ##
python train.py wandb_name_suffix="-ppp" steps=2501000 task=manipobj-v0 train_dataset_name=ppp-cube-noisy-v0-mv obs=ec_state_gen max_episode_steps=1000 env_kwargs.manipobj.num_cubes=3 agent=sgiql sgiql.alpha=0.1 sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.subgoal_k=50 sgiql.n_diffusion_samples=256 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 sgiql.value_competence_radius=-30

## EC-SGIQL AWR ##
python train.py wandb_name_suffix="-ppp_awr" steps=2501000 task=manipobj-v0 train_dataset_name=ppp-cube-noisy-v0-mv obs=ec_state_gen max_episode_steps=1000 env_kwargs.manipobj.num_cubes=3 agent=sgiql sgiql.alpha=0.1 sgiql.subgoal_policy_type=standard sgiql.beta=3.0 sgiql.subgoal_k=50 sgiql.subgoal_steps=25 sgiql.num_subgoals=32

## EC-IQL ##
python train.py wandb_name_suffix="-ppp" steps=2501000 task=manipobj-v0 train_dataset_name=ppp-cube-noisy-v0-mv obs=ec_state_gen max_episode_steps=1000 env_kwargs.manipobj.num_cubes=3 agent=iql iql.alpha=0.1

## HIQL ##
python train.py wandb_name_suffix="-ppp" steps=2501000 task=manipobj-v0 train_dataset_name=ppp-cube-noisy-v0-mv obs=state max_episode_steps=1000 env_kwargs.manipobj.num_cubes=3 agent=hiql hiql.alpha=0.1 hiql.beta=3.0 hiql.subgoal_k=50 hiql.subgoal_steps=25 hiql.num_subgoals=32

## IQL ##
python train.py wandb_name_suffix="-ppp" steps=2501000 task=manipobj-v0 train_dataset_name=ppp-cube-noisy-v0-mv obs=state max_episode_steps=1000 env_kwargs.manipobj.num_cubes=3 agent=iql iql.alpha=0.1

##########################


#### Stack-Cube (State) ####

## EC-SGIQL ##
python train.py wandb_name_suffix="-stack" steps=3001000 task=manipobj-v0 train_dataset_name=stack-cube-noisy-v0-mv obs=ec_state_gen max_episode_steps=1000 env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 agent=sgiql sgiql.alpha=0.05 sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.subgoal_k=50 sgiql.n_diffusion_samples=256 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 sgiql.value_competence_radius=-30

## EC-SGIQL AWR ##
python train.py wandb_name_suffix="-stack_awr" steps=3001000 task=manipobj-v0 train_dataset_name=stack-cube-noisy-v0-mv obs=ec_state_gen max_episode_steps=1000 env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 agent=sgiql sgiql.alpha=0.05 sgiql.subgoal_policy_type=standard sgiql.subgoal_k=50 sgiql.subgoal_steps=25 sgiql.num_subgoals=32

## EC-IQL ##
python train.py wandb_name_suffix="-stack" steps=3001000 task=manipobj-v0 train_dataset_name=stack-cube-noisy-v0-mv obs=ec_state_gen max_episode_steps=1000 env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 agent=iql iql.alpha=0.1

## HIQL ##
python train.py wandb_name_suffix="-stack" steps=3001000 task=manipobj-v0 train_dataset_name=stack-cube-noisy-v0-mv obs=state max_episode_steps=1000 env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 agent=hiql hiql.alpha=0.2 hiql.beta=3.0 hiql.subgoal_k=50 sgiql.subgoal_steps=25 sgiql.num_subgoals=32

## IQL ##
python train.py wandb_name_suffix="-stack" steps=3001000 task=manipobj-v0 train_dataset_name=stack-cube-noisy-v0-mv obs=state max_episode_steps=1000 env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 agent=iql iql.alpha=0.05

############################


#### PPP-Cube (Image) ####

## EC-SGIQL ##
python train.py wandb_name_suffix="-ppp" steps=2501000 task=manipobj-v0 train_dataset_name=ppp-cube-noisy-v0-mv obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 max_episode_steps=1000 env_kwargs.manipobj.num_cubes=3 agent=sgiql sgiql.alpha=0.2 sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.subgoal_k=50 sgiql.n_diffusion_samples=256 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 sgiql.value_competence_radius=-30 num_eval_episodes=50 eval_freq=250000

## EC-SGIQL AWR ##
python train.py wandb_name_suffix="-ppp_awr" steps=2501000 task=manipobj-v0 train_dataset_name=ppp-cube-noisy-v0-mv obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 max_episode_steps=1000 env_kwargs.manipobj.num_cubes=3 agent=sgiql sgiql.alpha=0.2 sgiql.subgoal_policy_type=standard sgiql.chamfer_target_weight=1.0 sgiql.chamfer_metric=l1 sgiql.beta=3.0 sgiql.subgoal_k=50 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 num_eval_episodes=50 eval_freq=250000

## EC-IQL ##
python train.py wandb_name_suffix="-ppp" steps=2501000 task=manipobj-v0 train_dataset_name=ppp-cube-noisy-v0-mv obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 max_episode_steps=1000 env_kwargs.manipobj.num_cubes=3 agent=iql iql.alpha=0.2 num_eval_episodes=50 eval_freq=250000

## HIQL ##
python train.py wandb_name_suffix="-ppp" steps=2501000 task=manipobj-v0 train_dataset_name=ppp-cube-noisy-v0-mv obs=vqvae multiview=true rep_model_checkpoint=visual_encoders/chkpts/vqvae-mv-ppp-cube-v0 max_episode_steps=1000 env_kwargs.manipobj.num_cubes=3 agent=hiql hiql.alpha=0.3 hiql.beta=3.0 hiql.subgoal_k=50 hiql.subgoal_steps=25 hiql.num_subgoals=32 num_eval_episodes=50 eval_freq=250000

## IQL ##
python train.py wandb_name_suffix="-ppp" steps=2501000 task=manipobj-v0 train_dataset_name=ppp-cube-noisy-v0-mv obs=vqvae multiview=true rep_model_checkpoint=visual_encoders/chkpts/vqvae-mv-ppp-cube-v0 max_episode_steps=1000 env_kwargs.manipobj.num_cubes=3 agent=iql iql.alpha=0.2 num_eval_episodes=50 eval_freq=250000

##########################


#### Scene (Image) ####

## EC-SGIQL ##
python train.py steps=1501000 task=visual-scene-v0 train_dataset_name=visual-scene-noisy-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-visual-scene-v0 max_episode_steps=1000 agent=sgiql sgiql.alpha=0.3 sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.subgoal_k=50 sgiql.n_diffusion_samples=64 sgiql.subgoal_steps=50 sgiql.num_subgoals=16 sgiql.value_competence_radius=-25 eval_freq=250000 num_eval_episodes=20 num_video_episodes=1

## EC-SGIQL AWR ##
python train.py steps=1501000 wandb_name_suffix="_awr" task=visual-scene-v0 train_dataset_name=visual-scene-noisy-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-visual-scene-v0 max_episode_steps=1000 agent=sgiql sgiql.alpha=0.3 sgiql.subgoal_policy_type=standard sgiql.beta=3.0 sgiql.subgoal_k=50 sgiql.subgoal_steps=50 sgiql.num_subgoals=16 sgiql.chamfer_target_weight=1.0 sgiql.chamfer_metric=l1 eval_freq=250000 num_eval_episodes=20 num_video_episodes=1

## EC-IQL ##
python train.py steps=1501000 task=visual-scene-v0 train_dataset_name=visual-scene-noisy-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-visual-scene-v0 max_episode_steps=1000 agent=iql iql.alpha=0.3 eval_freq=250000 num_eval_episodes=20 num_video_episodes=1

## HIQL ##
python train.py steps=1501000 task=visual-scene-v0 train_dataset_name=visual-scene-noisy-v0 obs=vqvae rep_model_checkpoint=visual_encoders/chkpts/vqvae-visual-scene-v0 max_episode_steps=1000 agent=hiql hiql.alpha=0.4 hiql.beta=3.0 hiql.subgoal_k=50 hiql.subgoal_steps=50 hiql.num_subgoals=16 eval_freq=250000 num_eval_episodes=20 num_video_episodes=1

## IQL ##
python train.py steps=1501000 task=visual-scene-v0 train_dataset_name=visual-scene-noisy-v0 obs=vqvae rep_model_checkpoint=visual_encoders/chkpts/vqvae-visual-scene-v0 max_episode_steps=1000 agent=iql iql.alpha=0.2 eval_freq=250000 num_eval_episodes=20 num_video_episodes=1

#######################


#### Push-Tetris (Image) ####

## EC-SGIQL ##
python train.py steps=1001000 task=pushtetris-v0 train_dataset_name=push-tetris-constrained_random-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 max_episode_steps=1000 env_kwargs.pushtetris.num_objects=3 agent=sgiql sgiql.alpha=0.1 sgiql.subgoal_policy_type=diffusion sgiql.subgoal_k=50 sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=64 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 sgiql.value_competence_radius=-20 eval_freq=250000 num_eval_episodes=100

## EC-SGIQL AWR ##
python train.py wandb_name_suffix="_awr" steps=1001000 task=pushtetris-v0 train_dataset_name=push-tetris-constrained_random-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 max_episode_steps=1000 env_kwargs.pushtetris.num_objects=3 agent=sgiql sgiql.alpha=0.1 sgiql.subgoal_policy_type=standard sgiql.beta=3.0 sgiql.subgoal_k=50 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 sgiql.chamfer_target_weight=1.0 sgiql.chamfer_metric=l1 eval_freq=250000 num_eval_episodes=100

## EC-IQL ##
python train.py steps=1001000 task=pushtetris-v0 train_dataset_name=push-tetris-constrained_random-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 max_episode_steps=1000 env_kwargs.pushtetris.num_objects=3 agent=iql iql.alpha=0.1 eval_freq=250000 num_eval_episodes=100

## HIQL ##
python train.py steps=1001000 task=pushtetris-v0 train_dataset_name=push-tetris-constrained_random-v0 obs=vqvae rep_model_checkpoint=visual_encoders/chkpts/vqvae-push-tetris-v0 max_episode_steps=1000 env_kwargs.pushtetris.num_objects=3 agent=hiql hiql.alpha=0.1 hiql.beta=3.0 hiql.subgoal_k=50 hiql.subgoal_steps=25 hiql.num_subgoals=32 hiql.chamfer_target_weight=1.0 hiql.chamfer_metric=l1 eval_freq=250000 num_eval_episodes=100

## IQL ##
python train.py steps=1001000 task=pushtetris-v0 train_dataset_name=push-tetris-constrained_random-v0 obs=vqvae rep_model_checkpoint=visual_encoders/chkpts/vqvae-push-tetris-v0 max_episode_steps=1000 env_kwargs.pushtetris.num_objects=3 agent=iql iql.alpha=0.1 eval_freq=250000 num_eval_episodes=100

#############################
