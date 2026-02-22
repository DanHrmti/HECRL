import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.modules.base import ImpalaEncoder


class Actor(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, a_dim, h_dim = cfg.obs_shape[-1], cfg.action_dim, cfg.arch_mlp['h_dim']

        if cfg.obs == 'rgb':
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        self.fc_mu = nn.Linear(h_dim, a_dim, device=device)

    def forward(self, x, g):
        x = torch.cat([x, g], dim=1)
        if self.gc_encoder is not None:
            x = x.float() / 255.0
            x = self.gc_encoder(x)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = self.fc_mu(x).tanh()
        return x


class QNetwork(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, a_dim, h_dim = cfg.obs_shape[-1], cfg.action_dim, cfg.arch_mlp['h_dim']

        if cfg.obs == 'rgb':
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim + a_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim + a_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.ln1 = nn.LayerNorm(h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.ln2 = nn.LayerNorm(h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        self.ln3 = nn.LayerNorm(h_dim, device=device)
        self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        self.ln4 = nn.LayerNorm(h_dim, device=device)
        self.fc_out = nn.Linear(h_dim, 1, device=device)

    def forward(self, x, a, g):
        x = torch.cat([x, g], dim=1)
        if self.gc_encoder is not None:
            x = x.float() / 255.0
            x = self.gc_encoder(x)
        x = torch.cat([x, a], dim=-1)
        x = self.ln1(F.gelu(self.fc1(x)))
        x = self.ln2(F.gelu(self.fc2(x)))
        x = self.ln3(F.gelu(self.fc3(x)))
        x = self.ln4(F.gelu(self.fc4(x)))
        x = self.fc_out(x)
        return x

class VNetwork(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, h_dim = cfg.obs_shape[-1], cfg.arch_mlp['h_dim']

        if cfg.obs == 'rgb':
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.ln1 = nn.LayerNorm(h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.ln2 = nn.LayerNorm(h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        self.ln3 = nn.LayerNorm(h_dim, device=device)
        self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        self.ln4 = nn.LayerNorm(h_dim, device=device)
        self.fc_out = nn.Linear(h_dim, 1, device=device)

    def forward(self, x, g):
        x = torch.cat([x, g], dim=1)
        if self.gc_encoder is not None:
            x = x.float() / 255.0
            x = self.gc_encoder(x)
        x = self.ln1(F.gelu(self.fc1(x)))
        x = self.ln2(F.gelu(self.fc2(x)))
        x = self.ln3(F.gelu(self.fc3(x)))
        x = self.ln4(F.gelu(self.fc4(x)))
        x = self.fc_out(x)
        return x

class GoalActor(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, a_dim, h_dim = cfg.obs_shape[-1], cfg.action_dim, cfg.arch_mlp['h_dim']

        if cfg.obs == 'rgb':
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        self.fc_out = nn.Linear(h_dim, obs_dim, device=device)

    def forward(self, x, g):
        x = torch.cat([x, g], dim=1)
        if self.gc_encoder is not None:
            x = x.float() / 255.0
            x = self.gc_encoder(x)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = self.fc_out(x)
        return x
