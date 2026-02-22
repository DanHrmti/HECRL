import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

from common.utils.rms_norm import RMSNormalizer

class Buffer():
	"""
	Replay buffer for training, based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg, env, capacity):
		self.cfg = cfg
		self.env = env
		self._device = torch.device('cuda:0')
		self._capacity = capacity
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=True, # NOTE: should be True in offline training
			compile=cfg.compile,
		)
		self.normalizer = RMSNormalizer(self.cfg.obs_shape[-1], disable=(self.cfg.obs=='rgb'))  # initialize RMS normalizer

		self._batch_size = cfg.batch_size * cfg.data_episode_length
		self._num_eps = 0
		if self.cfg.geom_sample_value or self.cfg.geom_sample_policy:
			self._geometric_dist = torch.distributions.Geometric(torch.tensor([1 - self.cfg.gamma], device=self._device))
		self.horizon = 1  # NOTE: fixed horizon=1, keeping for future use to sample trajectories

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity
	
	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps
	
	@property
	def obs_mean(self):
		return self.normalizer.mean
	
	@property
	def obs_std(self):
		return torch.sqrt(self.normalizer.var)

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, td):
		"""Initialize the replay buffer. Use the first tensordict to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in td.values()
		]) / len(td)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if 2.0*total_bytes < mem_free else 'cpu'  # 2.5 is the original value
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td) // (self.cfg.data_episode_length + 1) 
		td['episode'] = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64).repeat_interleave(self.cfg.data_episode_length + 1)
		if self._num_eps == 0:
			self._buffer = self._init(td[0].unsqueeze(0))
			self.normalizer.set_stats(td["obs"])  # set RMS normalizer stats once from offline data
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps
	
	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self.normalizer.update(td["obs"])  # update RMS normalizer stats
		self._num_eps += 1
		return self._num_eps
	
	def sample(self):
		# sample full episodes
		episodes = self._buffer.sample().view(-1, self.cfg.data_episode_length)
		# sample state indices
		idxs = torch.randint(self.cfg.data_episode_length - self.horizon, size=(self.cfg.batch_size,), device=self._storage_device)
		batch_idxs = torch.arange(self.cfg.batch_size, device=self._storage_device).unsqueeze(1)  # shape: (batch_size, 1)
		slice_idxs = idxs.unsqueeze(1) + torch.arange(self.horizon, device=self._storage_device).unsqueeze(0)
		# sample policy goal indices
		policy_goal_batch_idxs, policy_goal_idxs = self._sample_goal(idxs + self.horizon - 1, self.cfg.p_curgoal_policy, self.cfg.p_trajgoal_policy, self.cfg.p_randgoal_policy, self.cfg.geom_sample_policy)
		# sample value goal indices
		value_goal_batch_idxs, value_goal_idxs = self._sample_goal(idxs + self.horizon - 1, self.cfg.p_curgoal_value, self.cfg.p_trajgoal_value, self.cfg.p_randgoal_value, self.cfg.geom_sample_value)
		# retrieve data
		obs = episodes.get('obs')[batch_idxs, slice_idxs].contiguous().to(self._device, non_blocking=True)
		action = episodes.get('action')[batch_idxs, slice_idxs].contiguous().to(self._device, non_blocking=True)
		next_obs = episodes.get('obs')[batch_idxs, slice_idxs + 1].contiguous().to(self._device, non_blocking=True)
		value_goal = episodes.get('obs')[value_goal_batch_idxs, value_goal_idxs].unsqueeze(1).repeat(1, self.horizon, *[1] * len(self.cfg.obs_shape)).contiguous().to(self._device, non_blocking=True)
		policy_goal = episodes.get('obs')[policy_goal_batch_idxs, policy_goal_idxs].unsqueeze(1).repeat(1, self.horizon, *[1] * len(self.cfg.obs_shape)).contiguous().to(self._device, non_blocking=True)
		reward, done = self._compute_reward(episodes, idxs, batch_idxs, slice_idxs, value_goal_batch_idxs, value_goal_idxs)
		if self.cfg.agent in ['sgiql', 'hiql']:
			subgoal_idxs = torch.minimum(slice_idxs + self.cfg.subgoal_k, policy_goal_idxs.unsqueeze(1).repeat(1, self.horizon))
			policy_subgoal = episodes.get('obs')[batch_idxs, subgoal_idxs].contiguous().to(self._device, non_blocking=True)
		# save in tensordict
		td = TensorDict(
            obs=self.normalizer.normalize(obs),
			action=action,
			next_obs=self.normalizer.normalize(next_obs),
            reward=reward,
            done=done,
            value_goal=self.normalizer.normalize(value_goal),
            policy_goal=self.normalizer.normalize(policy_goal),
            batch_size=(obs.shape[0], obs.shape[1]),
            device=self._device,
        )
		if self.cfg.agent in ['sgiql', 'hiql']:
			td["policy_subgoal"] = self.normalizer.normalize(policy_subgoal)
		return td.squeeze(1)  # NOTE: this is to remove the extra 'horizon' dimension when self.horizon=1
	
	def _sample_goal(self, idxs, p_curgoal, p_trajgoal, p_randgoal, geom_sample):
		# random goals
		randgoal_idxs = torch.randint(self.cfg.data_episode_length - 1, size=(self.cfg.batch_size,), device=self._storage_device)
		randgoal_batch_idxs = torch.randint(self.cfg.batch_size, size=(self.cfg.batch_size,), device=self._storage_device)

		# goals from the same episode
		if geom_sample:
			offsets = self._geometric_dist.sample((self.cfg.batch_size,)).to(self._storage_device, non_blocking=True)
			trajgoal_idxs = torch.minimum(idxs + offsets.squeeze(-1), torch.full_like(idxs, self.cfg.data_episode_length-1)).to(torch.int)
		else:
			distances = torch.rand(self.cfg.batch_size, device=self._storage_device)  # in [0, 1)
			trajgoal_idxs = torch.round(((idxs + 1) * distances + (self.cfg.data_episode_length - 1) * (1 - distances))).to(torch.int)
		
		# goals at the next state
		curgoal_idxs = idxs + 1

		batch_idxs = torch.arange(self.cfg.batch_size, device=self._storage_device)
		goal_batch_idxs = torch.where(batch_idxs >= self.cfg.batch_size * (1-p_randgoal), randgoal_batch_idxs, batch_idxs)
		
		goal_idxs = torch.where(batch_idxs < self.cfg.batch_size * p_curgoal, curgoal_idxs, trajgoal_idxs)
		goal_idxs = torch.where(batch_idxs >= self.cfg.batch_size * (1-p_randgoal), randgoal_idxs, goal_idxs)

		return goal_batch_idxs, goal_idxs

	def _compute_reward(self, episodes, idxs, batch_idxs, slice_idxs, goal_batch_idxs, goal_idxs):
		
		if self.cfg.reward == 'unsupervised':
			# self-supervised reward
			episode_id = episodes.get('episode')[batch_idxs.squeeze(1), idxs].unsqueeze(1).repeat(1, self.horizon).contiguous()
			goal_episode_id = episodes.get('episode')[goal_batch_idxs, idxs].unsqueeze(1).repeat(1, self.horizon).contiguous()
			goal_idxs = goal_idxs.unsqueeze(1).repeat(1, self.horizon).contiguous()
			reward = torch.where((episode_id == goal_episode_id) & (slice_idxs+1 == goal_idxs), 0.0, -1.0).unsqueeze(-1).contiguous().to(self._device, non_blocking=True)
			done = torch.where(reward == 0, 1, 0).unsqueeze(-1).contiguous().to(self._device, non_blocking=True)

		else:
			if 'manipobj' in self.cfg.task:
				qpos = episodes.get('qpos')[batch_idxs, slice_idxs+1].contiguous().to(self._device, non_blocking=True)
				goal_qpos = episodes.get('qpos')[goal_batch_idxs, goal_idxs].unsqueeze(1).repeat(1, self.horizon, 1).contiguous().to(self._device, non_blocking=True)
				# reshape horizon to be part of the batch size
				qpos = qpos.view(self.cfg.batch_size * self.horizon, *qpos.shape[2:])
				goal_qpos = goal_qpos.view(self.cfg.batch_size * self.horizon, *goal_qpos.shape[2:])
				# compute reward
				reward = self.env.unwrapped.compute_reward_batch(qpos, goal_qpos)
				# reshape reward back to (bs, horizon, 1)
				reward = reward.view(self.cfg.batch_size, self.horizon, 1)
				done = torch.where(reward == 0, 1, 0).unsqueeze(-1).contiguous().to(self._device, non_blocking=True)
			
			elif 'scene' in self.cfg.task:
				qpos = episodes.get('qpos')[batch_idxs, slice_idxs+1].contiguous().to(self._device, non_blocking=True)
				goal_qpos = episodes.get('qpos')[goal_batch_idxs, goal_idxs].unsqueeze(1).repeat(1, self.horizon, 1).contiguous().to(self._device, non_blocking=True)
				button_states = episodes.get('button_states')[batch_idxs, slice_idxs+1].contiguous().to(self._device, non_blocking=True)
				goal_button_states = episodes.get('button_states')[goal_batch_idxs, goal_idxs].unsqueeze(1).repeat(1, self.horizon, 1).contiguous().to(self._device, non_blocking=True)
				# reshape horizon to be part of the batch size
				qpos = qpos.view(self.cfg.batch_size * self.horizon, *qpos.shape[2:])
				goal_qpos = goal_qpos.view(self.cfg.batch_size * self.horizon, *goal_qpos.shape[2:])
				button_states = button_states.view(self.cfg.batch_size * self.horizon, *button_states.shape[2:])
				goal_button_states = goal_button_states.view(self.cfg.batch_size * self.horizon, *goal_button_states.shape[2:])
				# compute reward
				reward = self.env.unwrapped.compute_reward_batch(qpos, goal_qpos, button_states, goal_button_states)
				# reshape reward back to (bs, horizon, 1)
				reward = reward.view(self.cfg.batch_size, self.horizon, 1)
				done = torch.where(reward == 0, 1, 0).unsqueeze(-1).contiguous().to(self._device, non_blocking=True)

			elif 'pushtetris' in self.cfg.task:
				obs = episodes.get('image_bitmasks')[batch_idxs, slice_idxs+1].contiguous().to(self._device, non_blocking=True)
				goal = episodes.get('image_bitmasks')[goal_batch_idxs, goal_idxs].unsqueeze(1).repeat(1, self.horizon, 1, 1).contiguous().to(self._device, non_blocking=True)
				# reshape horizon to be part of the batch size
				obs = obs.view(self.cfg.batch_size * self.horizon, *obs.shape[2:])
				goal = goal.view(self.cfg.batch_size * self.horizon, *goal.shape[2:])
				# compute reward
				reward = self.env.unwrapped.compute_reward_batch(obs, goal)
				# reshape reward back to (bs, horizon, 1)
				reward = reward.view(self.cfg.batch_size, self.horizon, 1)
				
			else:
				raise NotImplementedError(f"A supervised reward for task: {self.cfg.task} has not been defined.")
			
		done = done if self.cfg.done_signal else torch.zeros_like(reward)

		return reward, done
	