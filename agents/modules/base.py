import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResnetStack(nn.Module):
	def __init__(self, c_in, c, device=None):
		super().__init__()
		
		self.conv_in = nn.Conv2d(c_in, c, kernel_size=3, stride=1, padding=1, device=device)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.conv1 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, device=device)
		self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, device=device)

	def forward(self, x):
		x = self.conv_in(x)
		x_in = self.maxpool(x)
		# residual block
		x = self.conv1(F.relu(x_in))
		x = self.conv2(F.relu(x)) + x_in
		return x


class ImpalaEncoder(nn.Module):
	def __init__(self, cfg, device=None):
		super().__init__()

		channels, out_dim = cfg.arch_mlp['conv_channels'], cfg.arch_mlp['h_dim']
		c = [6, *channels]  # 6 for 3 input channels of state and goal images
		
		self.resnet = nn.Sequential(*[ResnetStack(c[i], c[i+1], device) for i in range(len(c)-1)])

		with torch.no_grad():
			dummy_input = torch.randn(1, 6, 64, 64, device=device)  # NOTE: adjust based on expected input size
			conv_out_flat_dim = self.resnet(dummy_input).view(-1).shape[0]

		self.linear = nn.Linear(conv_out_flat_dim, out_dim, device=device)

	def forward(self, x):
		x = F.relu(self.resnet(x))
		x = F.gelu(self.linear(x.view(*x.shape[:-3], -1)))
		return x


class EntityAttention(nn.Module):
	"""
	Entity-based multi-head masked attention layer with output projection
	"""
	def __init__(self, embed_dim, n_head, dropout=0.0, linear_bias=False, attn_type='self', device=None):
		super().__init__()
		assert embed_dim % n_head == 0
		assert attn_type in ['hybrid', 'cross', 'self']
		self.attn_type = attn_type

		self.n_head = n_head

		# key, query, value projections for all heads
		self.key = nn.Linear(embed_dim, embed_dim, bias=linear_bias, device=device)
		self.query = nn.Linear(embed_dim, embed_dim, bias=linear_bias, device=device)
		self.value = nn.Linear(embed_dim, embed_dim, bias=linear_bias, device=device)
		# regularization
		self.attn_drop = nn.Dropout(dropout)
		# output projection
		self.proj = nn.Linear(embed_dim, embed_dim, bias=linear_bias, device=device)
		
	def forward(self, x, c=None, mask=None):
		B, N, C = x.size()  # batch size, n_entities, embedding dimensionality (embed_dim)

		query_input = x
		if self.attn_type == 'hybrid':
			key_value_input = torch.cat([x, c], dim=1)
			key_value_N = key_value_input.shape[1]
		elif self.attn_type == 'cross':
			key_value_input = c
			key_value_N = key_value_input.shape[1]
		else:   # self.attn_type == 'self'
			key_value_input = x
			key_value_N = N

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q = self.query(query_input).view(B, N, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N, hs)
		k = self.key(key_value_input).view(B, key_value_N, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, key_value_N, hs)
		v = self.value(key_value_input).view(B, key_value_N, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, key_value_N, hs)

		# attention: (B, nh, N, hs) x (B, nh, hs, N) -> (B, nh, N, N)
		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N, key_value_N)
		if mask is not None:
			mask = mask.unsqueeze(1).expand(-1, self.n_head, -1, -1)
			att = att.masked_fill(mask, float('-inf'))
		att = F.softmax(att, dim=-1)
		att = self.attn_drop(att)
		y = att @ v  # (B, nh, N, key_value_N) x (B, nh, key_value_N, hs) -> (B, nh, N, hs)
		
		y = y.transpose(1, 2).contiguous().view(B, N, C)  # re-assemble all head outputs side by side

		# output projection
		y = self.proj(y)

		return y


class EITblock(nn.Module):
	def __init__(self, embed_dim, h_dim, n_head, dropout=0.1, attn_type='self', adaln=False, device=None):
		super().__init__()
		self.attn_type = attn_type
		self.adaln = adaln

		self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=(not adaln), eps=1e-6, device=device)
		self.ln2 = nn.LayerNorm(embed_dim, elementwise_affine=(not adaln), eps=1e-6, device=device)
		if self.attn_type != 'self':
			self.ln_c = nn.LayerNorm(embed_dim, eps=1e-6, device=device)
		
		if self.adaln:
			self.adaln_modulation = nn.Linear(embed_dim, 6 * embed_dim, device=device)

		self.attn = EntityAttention(embed_dim, n_head, dropout, False, attn_type, device=device)

		self.mlp = nn.Sequential(
			nn.Linear(embed_dim, h_dim, device=device),
			nn.GELU(),
			nn.Linear(h_dim, h_dim, device=device),
			nn.GELU(),
			nn.Linear(h_dim, embed_dim, device=device),
		)

		# weight initialization
		self.apply(self._init_weights)
		if self.adaln:
			# zero-out adaLN modulation layers
			nn.init.constant_(self.adaln_modulation.weight, 0)
			nn.init.constant_(self.adaln_modulation.bias, 0)


	def forward(self, x, c=None, adaln_c=None, mask=None):
		
		if self.attn_type != 'self':
			c = self.ln_c(c)

		if self.adaln:
			shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(adaln_c).chunk(6, dim=-1)
			x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.ln1(x), shift_msa, scale_msa), c, mask)
			x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.ln2(x), shift_mlp, scale_mlp))
		else:
			x = x + self.attn(self.ln1(x), c, mask)
			x = x + self.mlp(self.ln2(x))

		return x

	def _init_weights(self, module):
		std = 0.02
		if isinstance(module, nn.Linear):
			nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				nn.init.zeros_(module.bias)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
