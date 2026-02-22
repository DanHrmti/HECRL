import torch


class RMSNormalizer:
    def __init__(self, shape, epsilon=1e-6, disable=False):
        """
        Calculates the running mean and std (RMS) of a data stream for normalization purposes.
        """
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon
        self.epsilon = epsilon if not disable else 0
        self.disable = disable

    def normalize(self, obs):
        if self.disable:
            return obs
        else:
            return torch.clip((obs - self.mean.to(obs.device)) / torch.sqrt(self.var.to(obs.device) + self.epsilon), -5, 5)

    def unnormalize(self, obs):
        if self.disable:
            return obs
        else:
            return (obs * torch.sqrt(self.var.to(obs.device) + self.epsilon)) + self.mean.to(obs.device)
    
    def update(self, obs):
        if self.disable:
            return
        else:
            obs = obs.view(-1, obs.shape[-1])
            batch_mean = torch.mean(obs, dim=0)
            batch_var = torch.var(obs, dim=0)
            batch_count = obs.shape[0]
            self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def set_stats(self, obs):
        if self.disable:
            return
        else:
            obs = obs.view(-1, obs.shape[-1])
            self.mean = torch.mean(obs, dim=0)
            self.var = torch.var(obs, dim=0)
            self.count = obs.shape[0]
        