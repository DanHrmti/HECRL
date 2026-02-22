import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule

from common.utils.chamfer import chamfer_loss


class SGIQL(nn.Module):

    def __init__(self, cfg, Actor, GoalActor, QNetwork, VNetwork):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        # hyperparameters
        self.tau = self.cfg.sgiql['tau']
        self.expectile = self.cfg.sgiql['expectile']
        self.alpha = self.cfg.sgiql['alpha']
        self.beta = self.cfg.sgiql['beta']
        self.grad_clip_norm = self.cfg.grad_clip_norm
        self.subgoal_steps = self.cfg.sgiql["subgoal_steps"]
        self.num_subgoals = self.cfg.sgiql["num_subgoals"]
        self.disable_subgoal = self.cfg.sgiql["disable_subgoal"]
        self.subgoal_policy_type = self.cfg.sgiql["subgoal_policy_type"]
        self.n_diffusion_samples = self.cfg.sgiql["n_diffusion_samples"]
        self.value_competence_radius = self.cfg.sgiql["value_competence_radius"]
        self.load_pretrained_goal_pi = self.cfg.sgiql["load_pretrained_goal_pi"]
        self.filter_subgoals = self.cfg.sgiql["filter_subgoals"]
        self.debug_subgoals = self.cfg.sgiql["debug_subgoals"]

        # actor
        self.pi = Actor(self.cfg, self.device)
        
        self.pi_detach = Actor(self.cfg, self.device)
        from_module(self.pi).data.to_module(self.pi_detach)  # Copy params to pi_detach without grad

        # goal actor
        self.goal_pi = GoalActor(self.cfg, self.device)
        
        self.goal_pi_detach = GoalActor(self.cfg, self.device)
        from_module(self.goal_pi).data.to_module(self.goal_pi_detach)  # Copy params to goal_pi_detach without grad
        
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

        # pretrained weight initialization
        if self.cfg.train_from_checkpoint:
            self.load(self.cfg.checkpoint)
        if self.load_pretrained_goal_pi:
            print(f"Loading pretrained goal actor from {self.cfg.sgiql['goal_pi_chkpt']}")
            goal_pi_state_dict = torch.load(self.cfg.sgiql["goal_pi_chkpt"])
            self.goal_pi.load_state_dict(goal_pi_state_dict["goal_pi"])
        
        # optimizers
        self.actor_optimizer = optim.Adam(
            list(self.pi.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
        )

        self.goal_actor_optimizer = optim.Adam(
            list(self.goal_pi.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
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
            if self.subgoal_policy_type != 'diffusion':
                self._update_goal_pi = torch.compile(self._update_goal_pi, mode=mode)
            else:
                self._update_goal_pi_diffuser = torch.compile(self._update_goal_pi_diffuser, mode=mode)
            self._act = torch.compile(self._act, mode=mode)

        if self.cfg.cudagraphs:
            self._update_q = CudaGraphModule(self._update_q, in_keys=[], out_keys=[], warmup=5)
            self._update_v = CudaGraphModule(self._update_v, in_keys=[], out_keys=[], warmup=5)
            self._update_pi = CudaGraphModule(self._update_pi, in_keys=[], out_keys=[], warmup=5)
            if self.subgoal_policy_type != 'diffusion':
                self._update_goal_pi = CudaGraphModule(self._update_goal_pi, in_keys=[], out_keys=[], warmup=5)
            else:
                self._update_goal_pi_diffuser = CudaGraphModule(self._update_goal_pi_diffuser, in_keys=[], out_keys=[], warmup=5)
            self._act = CudaGraphModule(self._act, in_keys=[], out_keys=[], warmup=5)

    @torch.no_grad()
    def act(self, td):
        td = td.to(self.device, non_blocking=True)
        
        if td["step"] == 0:
            self._subgoals_executed = 0
            if self.debug_subgoals:
                print("\n\nNew Episode")

        if td["step"] % self.subgoal_steps == 0:

            if self.debug_subgoals:
                print(f'Step {td["step"]}')
                print(f'obs->goal value:     {self.vf_detach(td["obs"], td["goal"]).int().item()}')

            if self._subgoals_executed >= self.num_subgoals or self.disable_subgoal:
                self._subgoal = td["goal"]
            else:
                if self.subgoal_policy_type == "diffusion":
                    # sample a batch of subgoals from the diffusion model
                    obs, goal = td["obs"].clone().repeat(self.n_diffusion_samples, 1, 1), td["goal"].clone().repeat(self.n_diffusion_samples, 1, 1)
                    subgoal_samples = self.goal_pi_detach(obs, goal)
                    if self.cfg.obs in ['state', 'vqvae']:
                        obs, goal, subgoal_samples = obs.squeeze(1), goal.squeeze(1), subgoal_samples.squeeze(1)
                    subgoal_values = self.vf_detach(subgoal_samples, goal)

                    # filter for subgoals that are a "competence radius" (quantified by the value function) away from the current state
                    # NOTE: constants for the below value constraint should be chosen relative to the average value recorded during training
                    if self.filter_subgoals:
                        obs_to_subgoal_values = self.vf_detach(obs, subgoal_samples)
                        subgoal_values[obs_to_subgoal_values < self.value_competence_radius] = -torch.inf
                    
                    # choose subgoal that maximizes value while satisfying above constraints
                    subgoal_value, argmax_idx = torch.max(subgoal_values, dim=0)
                    self._subgoal = subgoal_samples[argmax_idx]

                    # if the current state is closer to the goal than the sampled subgoal, just go to the goal
                    if self.filter_subgoals:
                        cur_value = self.vf_detach(td["obs"], td["goal"])
                        if cur_value > subgoal_value:
                            self._subgoal = td["goal"]
                            if self.debug_subgoals:
                                print("Current state is closer to the goal than the sampled subgoal -- subgoal <- goal")

                    if self.debug_subgoals:
                        print(f"subgoal->goal value: {subgoal_values[argmax_idx].int().item()}")
                        if self.filter_subgoals:
                            print(f"obs->subgoal value:  {obs_to_subgoal_values[argmax_idx].int().item()}")
                            print(f"obs->subgoal sample values range:  [{obs_to_subgoal_values.min().int().item()}, {obs_to_subgoal_values.max().int().item()}]")

                else:
                    self._subgoal = self.goal_pi_detach(td["obs"], td["goal"])
            self._subgoals_executed += 1

        td["subgoal"] = self._subgoal

        td_out = self._act(td)

        action = td_out["action"]

        return action, {"subgoal": self._subgoal.detach().clone()}
    
    @torch.no_grad()
    def _act(self, td):
        obs = td["obs"]
        subgoal = td["subgoal"]

        action = self.pi_detach(obs, subgoal)
        
        return TensorDict(action=action.detach(), subgoal=subgoal.detach())
    
    def update(self, batch, step):
        # update value
        info = self._update_v(batch)

        # update q-value
        info.update(self._update_q(batch))

        # update actor
        info.update(self._update_pi(batch))

        # update goal actor
        if not self.load_pretrained_goal_pi:
            if self.subgoal_policy_type == 'diffusion':
                info.update(self._update_goal_pi_diffuser(batch))
            else:
                info.update(self._update_goal_pi(batch))
        
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
    
    def _update_goal_pi(self, batch):
        # NOTE: different from OGBench in that uses different hyperparameter (beta) for the high-level policy
        obs = batch["obs"]
        subgoal = batch["policy_subgoal"]
        goal = batch["policy_goal"]

        self.goal_actor_optimizer.zero_grad()

        goal_pi_out = self.goal_pi(obs, goal)

        ####### AWR #######
        v_current = self.vf_detach(obs, goal)
        v_subgoal = self.vf_detach(subgoal, goal)
        adv = v_subgoal - v_current
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100.0)
        if self.cfg.obs in ['dlp']:
            exp_adv = exp_adv / 100.0  # NOTE: normalization
            bc_losses = chamfer_loss(goal_pi_out, subgoal, self.cfg)
        else:
            bc_losses = torch.mean((goal_pi_out - subgoal)**2, dim=(-2, -1) if self.cfg.obs in ['ec_state', 'ec_state_gen'] else -1)
        goal_pi_loss = torch.mean(exp_adv * bc_losses)
        ###################

        goal_pi_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.goal_pi.parameters(), self.grad_clip_norm)
        self.goal_actor_optimizer.step()

        return TensorDict(goal_pi_loss=goal_pi_loss.detach(), goal_pi_adv=adv.mean().detach(), goal_pi_adv_max=adv.max().detach(), goal_pi_adv_min=adv.min().detach(), goal_pi_gradnorm=grad_norm)  # AWR
    
    def _update_goal_pi_diffuser(self, batch):
        obs = batch["obs"]
        subgoal = batch["policy_subgoal"]
        goal = batch["policy_goal"]

        B = obs.shape[0]

        self.goal_actor_optimizer.zero_grad()
        
        cond = {0: obs.view(B, -1).detach(), 2: goal.view(B, -1).detach()}  # 0: obs, 2: goal
        x = torch.cat([obs.unsqueeze(1), subgoal.unsqueeze(1), goal.unsqueeze(1)], dim=1).view(B, 3, -1)
        goal_pi_loss, _ = self.goal_pi.diffusion.loss(*(x, cond))

        goal_pi_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.goal_pi.parameters(), self.grad_clip_norm)
        self.goal_actor_optimizer.step()

        return TensorDict(goal_pi_diffusion_loss=goal_pi_loss.detach(), goal_pi_diffusion_gradnorm=grad_norm)
    
    def save(self, fp, obs_mean=0, obs_std=1):
        """
        Save state dict of the agent to filepath.

        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({
            "pi": self.pi.state_dict(),
            "goal_pi": self.goal_pi.state_dict(),
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
        if self.cfg.sgiql["load_pretrained_goal_pi"]:
            goal_pi_state_dict = torch.load(self.cfg.sgiql["goal_pi_chkpt"])
            self.goal_pi.load_state_dict(goal_pi_state_dict["goal_pi"])
        else:
            self.goal_pi.load_state_dict(state_dict["goal_pi"])
        self.vf.load_state_dict(state_dict["vf"])
        self.qnet_params.load_state_dict(state_dict["qfs"])
        self.qnet_target_params = self.qnet_params.data.clone()
        self.qnet_params.to_module(self.qnet)
        self.obs_mean = state_dict["obs_mean"]
        self.obs_std = state_dict["obs_std"]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    