import torch.nn as nn

from agents.diffuser.diffusion import GaussianDiffusion
from agents.diffuser.transformer_diffuser import TransformerDiffuser
from agents.diffuser.sampler import GoalConditionedSampler


class SubgoalDiffuser(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        self.cfg = cfg
        model = TransformerDiffuser(self.cfg)
        observation_dim = self.cfg.obs_shape[0]*self.cfg.obs_shape[1] if self.cfg.obs in ["ec_state", "ec_state_gen", "dlp"] else self.cfg.obs_shape[0]
        self.diffusion = GaussianDiffusion(model, horizon=3, observation_dim=observation_dim, n_timesteps=cfg.sgiql['n_diffusion_steps']).to(device)
        self.sampler = GoalConditionedSampler(self.diffusion, None, None)

    def forward(self, entities, goal_entities):
        B, N, D = entities.shape
        conditions = {0: entities.view(B, -1), 2: goal_entities.view(B, -1)}
        return self.sampler(conditions).view(B, N, D)
    