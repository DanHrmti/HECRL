#!/bin/bash


#### PPP-Cube (State) ####

## EC-SGIQL ##
python evaluate.py eval_variant_name='3 Cubes EC-SGIQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-SGIQL AWR ##
python evaluate.py eval_variant_name='3 Cubes EC-SGIQL AWR State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=standard sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-SGIQL Max Value Ablation ##
python evaluate.py eval_variant_name='3 Cubes EC-SGIQL State Max Value Ablation' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.filter_subgoals=false sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-SGIQL Random Sample Ablation ##
python evaluate.py eval_variant_name='3 Cubes EC-SGIQL State Random Sample Ablation' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=1 sgiql.filter_subgoals=false sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-IQL ##
python evaluate.py eval_variant_name='3 Cubes EC-IQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5  # SGIQL with disabled subgoals is simply IQL

## HIQL ##
python evaluate.py eval_variant_name='3 Cubes HIQL' task=manipobj-v0 obs=state env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=hiql hiql.subgoal_steps=25 hiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## IQL ##
python evaluate.py eval_variant_name='3 Cubes IQL' task=manipobj-v0 obs=state env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=iql eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

##########################


#### Stack-Cube (State) ####

## EC-SGIQL ##
python evaluate.py eval_variant_name='3 Cubes Stack EC-SGIQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5

## EC-SGIQL AWR ##
python evaluate.py eval_variant_name='3 Cubes Stack EC-SGIQL AWR State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=standard sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5

## EC-SGIQL Max Value Ablation ##
python evaluate.py eval_variant_name='3 Cubes Stack EC-SGIQL State Max Value Ablation' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.filter_subgoals=false sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5

## EC-SGIQL Random Sample Ablation ##
python evaluate.py eval_variant_name='3 Cubes Stack EC-SGIQL State Random Sample Ablation' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=1 sgiql.filter_subgoals=false sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5

## EC-IQL ##
python evaluate.py eval_variant_name='3 Cubes Stack EC-IQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5  # SGIQL with disabled subgoals is simply IQL

## HIQL ##
python evaluate.py eval_variant_name='3 Cubes Stack HIQL State' task=manipobj-v0 obs=state env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=hiql hiql.subgoal_steps=25 hiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5

## IQL ##
python evaluate.py eval_variant_name='3 Cubes Stack IQL State' task=manipobj-v0 obs=state env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=iql eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5

###########################


#### PPP-Cube (Image) ####

## EC-SGIQL ##
python evaluate.py eval_variant_name='3 Cubes EC-SGIQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-SGIQL AWR ##
python evaluate.py eval_variant_name='3 Cubes EC-SGIQL AWR Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=standard sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-SGIQL Max Value Ablation ##
python evaluate.py eval_variant_name='3 Cubes EC-SGIQL Max Value Ablation Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.filter_subgoals=false sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-SGIQL Random Sample Ablation ##
python evaluate.py eval_variant_name='3 Cubes EC-SGIQL Random Sample Ablation Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=1 sgiql.filter_subgoals=false sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-IQL ##
python evaluate.py eval_variant_name='3 Cubes EC-IQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5  # SGIQL with disabled subgoals is simply IQL

## HIQL ##
python evaluate.py eval_variant_name='3 Cubes HIQL Image' task=manipobj-v0 obs=vqvae multiview=true rep_model_checkpoint=visual_encoders/chkpts/vqvae-mv-ppp-cube-v0 env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=hiql hiql.subgoal_steps=25 hiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## IQL ##
python evaluate.py eval_variant_name='3 Cubes IQL Image' task=manipobj-v0 obs=vqvae multiview=true rep_model_checkpoint=visual_encoders/chkpts/vqvae-mv-ppp-cube-v0 env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=iql eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

##########################


#### Scene (Image) ####

## EC-SGIQL ##
python evaluate.py eval_variant_name='Scene EC-SGIQL Image' task=visual-scene-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-visual-scene-v0 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.subgoal_steps=50 sgiql.num_subgoals=16 sgiql.n_diffusion_samples=64 sgiql.value_competence_radius=-25 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=50 num_video_episodes=1

## EC-SGIQL AWR ##
python evaluate.py eval_variant_name='Scene EC-SGIQL AWR Image' task=visual-scene-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-visual-scene-v0 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=standard sgiql.subgoal_steps=50 sgiql.num_subgoals=16 eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=50 num_video_episodes=1

## EC-SGIQL Max Value Ablation ##
python evaluate.py eval_variant_name='Scene EC-SGIQL Max Value Ablation Image' task=visual-scene-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-visual-scene-v0 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.subgoal_steps=50 sgiql.num_subgoals=16 sgiql.n_diffusion_samples=64 sgiql.filter_subgoals=false eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=50 num_video_episodes=1

## EC-SGIQL Random Sample Ablation ##
python evaluate.py eval_variant_name='Scene EC-SGIQL Random Sample Ablation Image' task=visual-scene-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-visual-scene-v0 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.subgoal_steps=50 sgiql.num_subgoals=16 sgiql.n_diffusion_samples=1 sgiql.filter_subgoals=false eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=50 num_video_episodes=1

## EC-IQL ##
python evaluate.py eval_variant_name='Scene EC-IQL Image' task=visual-scene-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-visual-scene-v0 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=50 num_video_episodes=1  # SGIQL with disabled subgoals is simply IQL

## HIQL ##
python evaluate.py eval_variant_name='Scene HIQL Image' task=visual-scene-v0 obs=vqvae rep_model_checkpoint=visual_encoders/chkpts/vqvae-visual-scene-v0 max_episode_steps=1000 agent=hiql hiql.subgoal_steps=50 hiql.num_subgoals=16 eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=50 num_video_episodes=1

## IQL ##
python evaluate.py eval_variant_name='Scene IQL Image' task=visual-scene-v0 obs=vqvae rep_model_checkpoint=visual_encoders/chkpts/vqvae-visual-scene-v0 max_episode_steps=1000 agent=iql eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=50 num_video_episodes=1

#######################


#### Push-Tetris (Image) ####

## EC-SGIQL ##
python evaluate.py eval_variant_name='3 Objects EC-SGIQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=64 sgiql.value_competence_radius=-20 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-SGIQL AWR ##
python evaluate.py eval_variant_name='3 Objects EC-SGIQL AWR Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=standard sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-SGIQL Max Value Ablation ##
python evaluate.py eval_variant_name='3 Objects EC-SGIQL Max Value Ablation Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=64 sgiql.filter_subgoals=false sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-SGIQL Random Sample Ablation ##
python evaluate.py eval_variant_name='3 Objects EC-SGIQL Random Sample Ablation Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=1 sgiql.filter_subgoals=false sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-IQL ##
python evaluate.py eval_variant_name='3 Objects EC-IQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5  # SGIQL with disabled subgoals is simply IQL

## HIQL ##
python evaluate.py eval_variant_name='3 Objects HIQL Image' task=pushtetris-v0 obs=vqvae rep_model_checkpoint=visual_encoders/chkpts/vqvae-push-tetris-v0 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=3 max_episode_steps=1000 agent=hiql hiql.subgoal_steps=25 hiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## IQL ##
python evaluate.py eval_variant_name='3 Objects IQL Image' task=pushtetris-v0 obs=vqvae rep_model_checkpoint=visual_encoders/chkpts/vqvae-push-tetris-v0 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=3 max_episode_steps=1000 agent=iql eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

#############################




#### Generalization PPP-Cube (State) ####

## EC-SGIQL ##
python evaluate.py eval_variant_name='1 Cube  EC-SGIQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=1 max_episode_steps=500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=16 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='2 Cubes EC-SGIQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=2 max_episode_steps=800 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=28 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='3 Cubes EC-SGIQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='4 Cubes EC-SGIQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=4 max_episode_steps=1200 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=40 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='5 Cubes EC-SGIQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=5 max_episode_steps=1500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=52 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='6 Cubes EC-SGIQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=6 max_episode_steps=2000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=72 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-IQL ##
python evaluate.py eval_variant_name='1 Cube  EC-IQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=1 max_episode_steps=500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='2 Cubes EC-IQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=2 max_episode_steps=800 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='3 Cubes EC-IQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='4 Cubes EC-IQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=4 max_episode_steps=1200 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='5 Cubes EC-IQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=5 max_episode_steps=1500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='6 Cubes EC-IQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.num_cubes=6 max_episode_steps=2000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

#########################################


### Generalization Stack-Cube (State) ###

## EC-SGIQL ##
python evaluate.py eval_variant_name='2 Cubes Stack EC-SGIQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=2 max_episode_steps=500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=16 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5
python evaluate.py eval_variant_name='4 Cubes Stack 2 EC-SGIQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.mode=task_stack env_kwargs.manipobj.num_cubes=4 max_episode_steps=1200 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=40 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5

## EC-IQL ##
python evaluate.py eval_variant_name='2 Cubes Stack EC-IQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.mode=task_stack_all env_kwargs.manipobj.num_cubes=2 max_episode_steps=500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5
python evaluate.py eval_variant_name='4 Cubes Stack 2 EC-IQL State' task=manipobj-v0 obs=ec_state_gen env_kwargs.manipobj.mode=task_stack env_kwargs.manipobj.num_cubes=4 max_episode_steps=1200 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=200 num_video_episodes=5

#########################################


#### Generalization PPP-Cube (Image) ####

## EC-SGIQL ##
python evaluate.py eval_variant_name='1 Cube  EC-SGIQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=-1 env_kwargs.manipobj.num_cubes=1 max_episode_steps=500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=16 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='2 Cubes EC-SGIQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=-1 env_kwargs.manipobj.num_cubes=2 max_episode_steps=800 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=28 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='3 Cubes EC-SGIQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=-1 env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='4 Cubes EC-SGIQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=24 env_kwargs.manipobj.num_cubes=4 max_episode_steps=1200 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=40 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='5 Cubes EC-SGIQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=28 env_kwargs.manipobj.num_cubes=5 max_episode_steps=1500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=52 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='6 Cubes EC-SGIQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=30 env_kwargs.manipobj.num_cubes=6 max_episode_steps=2000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=256 sgiql.value_competence_radius=-30 sgiql.subgoal_steps=25 sgiql.num_subgoals=72 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-IQL ##
python evaluate.py eval_variant_name='1 Cube  EC-IQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=-1 env_kwargs.manipobj.num_cubes=1 max_episode_steps=500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='2 Cubes EC-IQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=-1 env_kwargs.manipobj.num_cubes=2 max_episode_steps=800 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='3 Cubes EC-IQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=-1 env_kwargs.manipobj.num_cubes=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='4 Cubes EC-IQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=24 env_kwargs.manipobj.num_cubes=4 max_episode_steps=1200 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='5 Cubes EC-IQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=28 env_kwargs.manipobj.num_cubes=5 max_episode_steps=1500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='6 Cubes EC-IQL Image' task=manipobj-v0 obs=dlp multiview=true rep_model_checkpoint=visual_encoders/chkpts/dlp-mv-ppp-cube-v0 dlp_post_training_num_kp=30 env_kwargs.manipobj.num_cubes=6 max_episode_steps=2000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

#########################################


#### Generalization Push-Tetris (Image) ####

## EC-SGIQL ##
python evaluate.py eval_variant_name='1 Object  EC-SGIQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=-1 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=1 max_episode_steps=500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=64 sgiql.value_competence_radius=-10 sgiql.subgoal_steps=25 sgiql.num_subgoals=16 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='2 Objects EC-SGIQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=-1 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=2 max_episode_steps=800 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=64 sgiql.value_competence_radius=-15 sgiql.subgoal_steps=25 sgiql.num_subgoals=28 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='3 Objects EC-SGIQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=-1 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=64 sgiql.value_competence_radius=-20 sgiql.subgoal_steps=25 sgiql.num_subgoals=32 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='4 Objects EC-SGIQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=24 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=4 max_episode_steps=1200 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=64 sgiql.value_competence_radius=-20 sgiql.subgoal_steps=25 sgiql.num_subgoals=40 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='5 Objects EC-SGIQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=26 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=5 max_episode_steps=1500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=64 sgiql.value_competence_radius=-20 sgiql.subgoal_steps=25 sgiql.num_subgoals=52 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='6 Objects EC-SGIQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=28 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=6 max_episode_steps=2000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=64 sgiql.value_competence_radius=-20 sgiql.subgoal_steps=25 sgiql.num_subgoals=72 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='7 Objects EC-SGIQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=30 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=7 max_episode_steps=2000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.n_diffusion_steps=10 sgiql.n_diffusion_samples=64 sgiql.value_competence_radius=-20 sgiql.subgoal_steps=25 sgiql.num_subgoals=72 eval_checkpoint=output/train/<run_name>/models/latest.pt sgiql.load_pretrained_goal_pi=true sgiql.goal_pi_chkpt=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

## EC-IQL ##
python evaluate.py eval_variant_name='1 Object  EC-IQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=-1 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=1 max_episode_steps=500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='2 Objects EC-IQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=-1 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=2 max_episode_steps=800 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='3 Objects EC-IQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=-1 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=3 max_episode_steps=1000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='4 Objects EC-IQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=24 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=4 max_episode_steps=1200 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='5 Objects EC-IQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=26 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=5 max_episode_steps=1500 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='6 Objects EC-IQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=28 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=6 max_episode_steps=2000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5
python evaluate.py eval_variant_name='7 Objects EC-IQL Image' task=pushtetris-v0 obs=dlp rep_model_checkpoint=visual_encoders/chkpts/dlp-push-tetris-v0 dlp_post_training_num_kp=30 env_kwargs.pushtetris.reward_mode=chamfer env_kwargs.pushtetris.num_objects=7 max_episode_steps=2000 agent=sgiql sgiql.subgoal_policy_type=diffusion sgiql.disable_subgoal=true eval_checkpoint=output/train/<run_name>/models/latest.pt num_eval_episodes=100 num_video_episodes=5

############################################
