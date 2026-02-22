import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule


class IQL(nn.Module):

    def __init__(self, cfg, Actor, QNetwork, VNetwork):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        # hyperparameters
        self.tau = self.cfg.iql['tau']
        self.expectile = self.cfg.iql['expectile']
        self.alpha = self.cfg.iql['alpha']
        self.grad_clip_norm = self.cfg.grad_clip_norm

        # actor
        self.pi = Actor(self.cfg, self.device)
        
        self.pi_detach = Actor(self.cfg, self.device)
        from_module(self.pi).data.to_module(self.pi_detach)  # Copy params to pi_detach without grad
        
        # value
        self.vf = VNetwork(self.cfg, self.device)

        self.vf_detach = VNetwork(self.cfg, self.device)
        from_module(self.vf).data.to_module(self.vf_detach)  # Copy params to vf_detach without grad

        # q-value
        qf1 = QNetwork(self.cfg, self.device)
        qf2 = QNetwork(self.cfg, self.device)

        self.qnet_params = from_modules(qf1, qf2, as_module=True)
        self.qnet_target_params = self.qnet_params.data.clone()
        # discard params of net
        self.qnet = QNetwork(self.cfg, device="meta")
        self.qnet_params.to_module(self.qnet)

        # optimizers
        self.actor_optimizer = optim.Adam(
            list(self.pi.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
        )
        
        self.v_optimizer = optim.Adam(
            list(self.vf.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
        )

        self.q_optimizer = optim.Adam(
            self.qnet_params.values(include_nested=True, leaves_only=True),
            lr=self.cfg.lr,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
        )

        # torch.compile
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self._update_q = torch.compile(self._update_q, mode=mode)
            self._update_v = torch.compile(self._update_v, mode=mode)
            self._update_pi = torch.compile(self._update_pi, mode=mode)
            self._act = torch.compile(self._act, mode=mode)

        if self.cfg.cudagraphs:
            self._update_q = CudaGraphModule(self._update_q, in_keys=[], out_keys=[], warmup=5)
            self._update_v = CudaGraphModule(self._update_v, in_keys=[], out_keys=[], warmup=5)
            self._update_pi = CudaGraphModule(self._update_pi, in_keys=[], out_keys=[], warmup=5)
            self._act = CudaGraphModule(self._act, in_keys=[], out_keys=[], warmup=5)

    @torch.no_grad()
    def act(self, td):
        td_out = self._act(td.to(self.device, non_blocking=True))
        action = td_out["action"]
        return action, {}
    
    @torch.no_grad()
    def _act(self, td):
        obs = td["obs"]
        goal = td["goal"]
        action = self.pi_detach(obs, goal)
        return TensorDict(action=action.detach())
    
    def update(self, batch, step):
        # update value
        info = self._update_v(batch)

        # update q-value
        info.update(self._update_q(batch))

        # update actor
        info.update(self._update_pi(batch))
        
        # update the target networks
        # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
        self.qnet_target_params.lerp_(self.qnet_params.data, self.tau)
        
        return info.detach().mean()
    
    def _update_q(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]
        reward = batch["reward"]
        done = batch["done"]
        goal = batch["value_goal"]
        
        vf_next_target = self.vf_detach(next_obs, goal)
        next_q_value = reward.flatten() + (1 - done.flatten()) * self.cfg.gamma * vf_next_target.flatten()  # whether a "done" signal is recieved depends on config
        next_q_value = torch.clamp(next_q_value, min=-1/(1 - self.cfg.gamma), max=0)  # NOTE: clamp to possible range based on the reward, assumes reward in [-1, 0]

        qf_loss = torch.vmap(self._batched_qf, (0, None, None, None, None))(self.qnet_params, obs, action, goal, next_q_value)
        qf_loss = qf_loss.sum(0)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.qnet_params.values(include_nested=True, leaves_only=True), self.grad_clip_norm)
        self.q_optimizer.step()

        return TensorDict(qf_loss=qf_loss.detach(), qf_gradnorm=grad_norm)

    def _batched_qf(self, params, obs, action, goal, next_q_value=None):
            with params.to_module(self.qnet):
                vals = self.qnet(obs, action, goal)
                if next_q_value is not None:
                    loss_val = F.mse_loss(vals.view(-1), next_q_value)
                    return loss_val
                return vals
            
    def _update_v(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        goal = batch["value_goal"]

        qf_target = torch.vmap(self._batched_qf, (0, None, None, None))(self.qnet_target_params, obs, action, goal)
        min_qf_target = qf_target.min(0).values

        value = self.vf(obs, goal)
        
        value_loss = torch.mean(torch.abs(self.expectile - ((min_qf_target - value) < 0).float()) * (min_qf_target - value)**2)  # expectile loss
        
        self.v_optimizer.zero_grad()
        value_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.vf.parameters(), self.grad_clip_norm)
        self.v_optimizer.step()
        
        return TensorDict(vf_loss=value_loss.detach(), vf_gradnorm=grad_norm)

    def _update_pi(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        goal = batch["policy_goal"]

        # NOTE: different from OGBench in that uses qf1 and not min(qf1, qf2) for policy training
        self.actor_optimizer.zero_grad()
        with self.qnet_params.data[0].to_module(self.qnet):
            pi_out = self.pi(obs, goal)
            q_out = self.qnet(obs, pi_out, goal)

            q_loss = -q_out.mean() / (q_out.abs().mean().detach() + 1e-6)
            bc_loss = F.mse_loss(pi_out, action)
            
            pi_loss = q_loss + self.alpha * bc_loss

        pi_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        return TensorDict(pi_loss=pi_loss.detach(), pi_q_loss=q_loss.detach(), pi_bc_loss=bc_loss.detach(), pi_gradnorm=grad_norm, q_scale=q_out.abs().mean().detach())
    
    def save(self, fp, obs_mean=0, obs_std=1):
        """
        Save state dict of the agent to filepath.

        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({
            "pi": self.pi.state_dict(),
            "vf": self.vf.state_dict(),
            "qfs": self.qnet_params.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            }, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent for inference.

        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = torch.load(fp)
        self.pi.load_state_dict(state_dict["pi"])
        self.vf.load_state_dict(state_dict["vf"])
        self.qnet_params.load_state_dict(state_dict["qfs"])
        self.qnet_target_params = self.qnet_params.data.clone()
        self.qnet_params.to_module(self.qnet)
        self.obs_mean = state_dict["obs_mean"]
        self.obs_std = state_dict["obs_std"]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    