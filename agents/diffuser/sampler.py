import torch
import einops


def to_torch(x, dtype=None, device=None):
	dtype = dtype or torch.float
	device = device or 'cuda:0'
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
	return torch.tensor(x, dtype=dtype, device=device)


def apply_dict(fn, d, *args, **kwargs):
	return {
		k: fn(v, *args, **kwargs)
		for k, v in d.items()
	}


class GoalConditionedSampler:
    def __init__(self, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = lambda x : x
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        if len(conditions[0].shape) == 1:
            multi_input = False
        else:
            multi_input = True
        conditions = self._format_conditions(conditions, batch_size, multi_input=multi_input)
        
        samples = self.diffusion_model(conditions, sort_by_value=False, **self.sample_kwargs)
        
        normed_observations = samples.trajectories[:, :, self.action_dim:]
        return normed_observations[:, 1, :]

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size, multi_input=False):
        
        conditions = to_torch(conditions, dtype=torch.float32)
        if not multi_input:
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'd -> repeat d', repeat=batch_size,
            )
        return conditions
