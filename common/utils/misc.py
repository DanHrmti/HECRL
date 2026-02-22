import random

import numpy as np
import torch

from scipy.optimize import linear_sum_assignment


def set_seed(seed):
	"""Set seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def visualize_frame(cfg, env, reward, obs_mean, obs_std, info, goal_info, act_info):
    """
    Visualize the current frame of the environment.
    Handles multiview observations, goal frames, and generated subgoal frames if applicable.
    """
    
    frame = env.render({"obs_mean": obs_mean, "obs_std": obs_std})
    if len(frame.shape) == 4:
        frame = np.concatenate([frame[i] for i in range(len(frame))], axis=1)

    goal_frame = goal_info['goal_rendered']
    if len(goal_frame.shape) == 4:
        goal_frame = np.concatenate([goal_frame[i] for i in range(len(goal_frame))], axis=1)
    
    if cfg.agent == 'sgiql' and cfg.obs in ['dlp'] and not cfg.sgiql["disable_subgoal"]:
        goal_image = goal_info['goal_rendered']
        if len(goal_image.shape) == 3:
            goal_image = goal_image.reshape(1, *goal_image.shape)
        raw_goal_image = torch.from_numpy(goal_image).permute(0, 3, 1, 2).to(env.preprocess_info["device"])
        z_bg = env.preprocess_info["rep_model"].encode_all(raw_goal_image.to(torch.float32) / 255)["z_bg"]
        subgoal = act_info["subgoal"] * obs_std.to(env.preprocess_info["device"]) + obs_mean.to(env.preprocess_info["device"])
        subgoal = subgoal.reshape(2 if cfg.multiview else 1, -1, subgoal.shape[-1]).to(env.preprocess_info["device"])
        normalized_subgoal_image = env.preprocess_info["rep_model"].decode_all(
                                        z=subgoal[..., :2], z_scale=subgoal[..., 2:4], z_depth=subgoal[..., 4:5], obj_on=subgoal[..., 5],
                                        z_features=subgoal[..., 6:], z_bg=z_bg, noisy=False
                                    )["rec"]  # NOTE: 'dec_objects_trans' is foreground reconstruction
        subgoal_frame = np.moveaxis(normalized_subgoal_image.cpu().numpy() * 255, 1, -1).astype(np.uint8)
        if len(subgoal_frame) == 1:
            subgoal_frame = subgoal_frame.squeeze(0)
        else:
            subgoal_frame = np.concatenate([subgoal_frame[i] for i in range(len(subgoal_frame))], axis=1)
                            
        frame = np.concatenate([goal_frame, subgoal_frame, frame], axis=0)
    
    elif cfg.agent in ['hiql', 'sgiql'] and cfg.obs in ['vqvae'] and not cfg.sgiql["disable_subgoal"]:
        subgoal = torch.clamp(act_info["subgoal"] * obs_std.to(env.preprocess_info["device"]) + obs_mean.to(env.preprocess_info["device"]), -1, 1)
        subgoal = subgoal.reshape(2 if cfg.multiview else 1, 16, 8, 8).to(env.preprocess_info["device"])
        normalized_subgoal_image = env.preprocess_info["rep_model"].decode(subgoal)
        subgoal_frame = np.moveaxis(((normalized_subgoal_image.cpu().numpy() + 1 ) / 2)  * 255, 1, -1).astype(np.uint8)
        if len(subgoal_frame) == 1:
            subgoal_frame = subgoal_frame.squeeze(0)
        else:
            subgoal_frame = np.concatenate([subgoal_frame[i] for i in range(len(subgoal_frame))], axis=1)
                            
        frame = np.concatenate([goal_frame, subgoal_frame, frame], axis=0)
    
    else:
        frame = np.concatenate([goal_frame, frame], axis=0)

    return frame


def calc_num_entity_mods(cfg, env, obs, subgoal, obs_mean, obs_std):
    """Calculate the number of entity modifications between the current observation and the subgoal."""
    
    assert cfg.obs in ['state', 'ec_state', 'ec_state_gen']
    
    unnormalized_ob = obs.squeeze(0) * obs_std + obs_mean
    unnormalized_sg = subgoal.squeeze(0).cpu() * obs_std + obs_mean

    if cfg.obs == 'ec_state_gen':
        # Split into ids and positions
        entity_id_ob = unnormalized_ob[:, -9:]
        entity_id_sg = unnormalized_sg[:, -9:]
        entity_pos_ob = unnormalized_ob[:, :3]
        entity_pos_sg = unnormalized_sg[:, :3]

        # Compute pairwise distance matrix on ids
        # entity_id_ob: [N, 9], entity_id_sg: [N, 9]
        diff = entity_id_ob[:, None, :] - entity_id_sg[None, :, :]   # [N, N, 9]
        cost_matrix = torch.norm(diff, dim=-1)                       # [N, N]

        # Solve assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

        # Get matched values
        entity_pos_matched_ob = entity_pos_ob[row_ind]
        entity_pos_matched_sg = entity_pos_sg[col_ind]
    else:
        entity_pos_matched_ob = torch.cat([unnormalized_ob[:5].unsqueeze(0), unnormalized_ob[6:].reshape(-1, 5)], dim=0)[:, :3]
        entity_pos_matched_sg = torch.cat([unnormalized_sg[:5].unsqueeze(0), unnormalized_sg[6:].reshape(-1, 5)], dim=0)[:, :3]

    # Compute L2 distances in object positions
    dists = torch.norm(entity_pos_matched_ob - entity_pos_matched_sg, dim=1)
    
    # Count how many distances pass the threshold
    num_entity_mods = (dists > 0.04).sum().item()

    return num_entity_mods
