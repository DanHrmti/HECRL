import math

import torch
from torch import nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Sinusoidal Embedding for Diffusion Timesteps
# ------------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ------------------------------------------------------------------------------
# Entity-Based Attention Modules
# ------------------------------------------------------------------------------

class EntityAttention(nn.Module):
    """
    Entity-based multi-head masked self-attention layer.

    This module implements multi-head attention over entity representations.
    It supports three types of attention:
      - 'self': standard self-attention.
      - 'cross': attention over a conditioning input `c`.
      - 'hybrid': concatenates self and conditioning inputs.
    
    Args:
        n_embed (int): Dimensionality of the input embeddings.
        n_head (int): Number of attention heads.
        attn_pdrop (float): Dropout probability on attention weights.
        resid_pdrop (float): Dropout probability on the output projection.
        linear_bias (bool): If True, include bias terms in the linear projections.
        att_type (str): Type of attention: 'self', 'cross', or 'hybrid'.
    """
    def __init__(self, n_embed, n_head, attn_pdrop=0.1, resid_pdrop=0.1, linear_bias=False, att_type='self'):
        super().__init__()
        assert n_embed % n_head == 0
        assert att_type in ['hybrid', 'cross', 'self']
        self.att_type = att_type
        # Key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # Dropout layers for regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # Output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.n_head = n_head

    def forward(self, x, c=None):
        B, N, T, C = x.size()  # batch size, n_entities, sequence length, embedding dim
        # Determine inputs for query and key/value projections based on attention type.
        query_input = x
        if self.att_type == 'hybrid':
            key_value_input = torch.cat([x, c], dim=1)
            key_value_N = key_value_input.shape[1]
            key_value_T = key_value_input.shape[2]
        elif self.att_type == 'cross':
            key_value_input = c
            key_value_N = key_value_input.shape[1]
            key_value_T = key_value_input.shape[2]
        else:   # self-attention
            key_value_input = x
            key_value_N = N
            key_value_T = T

        # Compute projections and reshape for multi-head attention.
        k = self.key(key_value_input).view(B, key_value_N * key_value_T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(query_input).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(key_value_input).view(B, key_value_N * key_value_T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, N*T, key_value_N*key_value_T) x (B, nh, key_value_N*key_value_T, hs) -> (B, nh, N*T, hs)
        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)
        
        return y


# ------------------------------------------------------------------------------
# Feed-Forward Network (MLP)
# ------------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Feed-forward multi-layer perceptron (MLP) used within transformer blocks.

    This module is a two-layer fully-connected network with an activation function (GELU or ReLU)
    between the layers and dropout applied after the activation.

    Args:
        n_embed (int): Input and output embedding dimensionality.
        resid_pdrop (float): Dropout probability applied after activation.
        hidden_dim_multiplier (int): Factor to multiply `n_embed` for the hidden layer size.
        activation (str): Activation function to use ('gelu' or 'relu').
    
    Example:
        >>> mlp = MLP(n_embed=256, hidden_dim_multiplier=4)
        >>> x = torch.randn(32, 10, 256)
        >>> output = mlp(x)  # shape: (32, 10, 256)
    """
    def __init__(self, n_embed, resid_pdrop=0.1, hidden_dim_multiplier=4, activation='gelu'):
        super().__init__()
        self.fc_1 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed)
        if activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(True)
        self.proj = nn.Linear(hidden_dim_multiplier * n_embed, n_embed)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = self.dropout(self.proj(self.act(self.fc_1(x))))
        return x


# ------------------------------------------------------------------------------
# Final Adaptive Modulation Layers
# ------------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FinalLayer(nn.Module):
    """
    Final adaptive modulation layer.
    """
    def __init__(self, dim):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        # Zero-out modulation layers to start as identity
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias) 

    def forward(self, x, c):
        scale = self.gamma(c)
        shift = self.beta(c)
        x = modulate(self.ln_final(x), shift, scale)
        return x


# ------------------------------------------------------------------------------
# Adaptive Layer-Normalized Entity Interaction Transformer Block
# ------------------------------------------------------------------------------

class AdaLNEITBlock(nn.Module):
    """
    Adaptive Layer Normalized Entity Interaction Transformer (AdaLN-EIT) block.

    This block extends a standard transformer block by incorporating adaptive layer
    normalization. The normalization parameters are conditioned on an external input `c`
    and are used to modulate both the attention and MLP sub-layers.

    Args:
        n_embed (int): Embedding dimensionality.
        n_head (int): Number of attention heads.
        attn_pdrop (float): Dropout probability for attention weights.
        resid_pdrop (float): Dropout probability for residual connections.
        hidden_dim_multiplier (int): Multiplier for the hidden dimension in the MLP.
        activation (str): Activation function for the MLP ('gelu' or 'relu').
        att_type (str): Type of attention ('self', 'cross', or 'hybrid').
    """
    def __init__(self, n_embed, n_head, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 activation='gelu', att_type='self'):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed, elementwise_affine=False, eps=1e-6)
        self.attn = EntityAttention(n_embed, n_head, attn_pdrop, resid_pdrop, att_type=att_type)
        self.ln_2 = nn.LayerNorm(n_embed, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)
        self.gamma_1 = nn.Linear(n_embed, n_embed)
        self.beta_1 = nn.Linear(n_embed, n_embed)
        self.gamma_2 = nn.Linear(n_embed, n_embed)
        self.beta_2 = nn.Linear(n_embed, n_embed)
        self.scale_1 = nn.Linear(n_embed, n_embed)
        self.scale_2 = nn.Linear(n_embed, n_embed)

        nn.init.zeros_(self.gamma_1.weight)
        nn.init.zeros_(self.beta_1.weight)
        nn.init.zeros_(self.gamma_1.bias)
        nn.init.zeros_(self.beta_1.bias)  

        nn.init.zeros_(self.gamma_2.weight)
        nn.init.zeros_(self.beta_2.weight)
        nn.init.zeros_(self.gamma_2.bias)
        nn.init.zeros_(self.beta_2.bias)  

        nn.init.zeros_(self.scale_1.weight)
        nn.init.zeros_(self.scale_2.weight)
        nn.init.zeros_(self.scale_1.bias)
        nn.init.zeros_(self.scale_2.bias)  

    def forward(self, x, c):
        scale_msa = self.gamma_1(c)
        shift_msa = self.beta_1(c)
        scale_mlp = self.gamma_2(c)
        shift_mlp = self.beta_2(c)
        gate_msa = self.scale_1(c).unsqueeze(1)
        gate_mlp = self.scale_2(c).unsqueeze(1)
        x = self.attn(modulate(self.ln_1(x), shift_msa, scale_msa)) * gate_msa + x
        return self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)) * gate_mlp + x
    

# ------------------------------------------------------------------------------
# Adaptive Layer Normalized Entity Transformer (Top-Level Model)
# ------------------------------------------------------------------------------

class AdaLNEntityTransformer(nn.Module):
    """
    Adaptive Layer Normalized Entity Transformer.

    This is a full transformer model designed for entity-based data. It stacks multiple
    AdaLN-EIT blocks and applies either shared or per-entity positional embeddings.
    The network supports conditioning via external embeddings (such as action and temporal cues)
    which modulate the normalization layers in each block.

    Args:
        n_embed (int): Embedding dimensionality for entity features.
        n_head (int): Number of attention heads.
        n_layer (int): Number of transformer blocks.
        block_size (int): Maximum sequence length (number of time steps).
        output_dim (int): Dimensionality of the final output.
        attn_pdrop (float): Dropout probability for attention weights.
        resid_pdrop (float): Dropout probability for residual connections.
        hidden_dim_multiplier (int): Multiplier for the hidden dimension in MLPs.
        activation (str): Activation function for MLPs ('gelu' or 'relu').
    """
    def __init__(self, n_embed, n_head, n_layer, block_size, output_dim, attn_pdrop=0.1, resid_pdrop=0.1,
                 hidden_dim_multiplier=4, activation='gelu'):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embed))
        # Build the stacked transformer blocks.
        self.blocks = nn.Sequential(*[AdaLNEITBlock(n_embed, n_head, attn_pdrop,
                                                       resid_pdrop, hidden_dim_multiplier,
                                                       activation=activation)
                                      for _ in range(n_layer)])
        # Final modulation and output projection.
        self.ln_f = FinalLayer(n_embed)
        self.head = nn.Linear(n_embed, output_dim, bias=False)

        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layer = n_layer

        # Initialize weights.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, AdaLNEntityTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=std)

    def forward(self, x, action_embed, t_embed):
        b, n, t, f = x.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        assert f == self.n_embed, "invalid entity feature dim"
        if t_embed is not None:
            c = action_embed + t_embed.unsqueeze(1)  # combine conditioning signals
        else:
            c = action_embed
        position_embeddings = self.pos_emb[:, None, :t, :]  # (1, 1, t, n_embed)
        x = x + position_embeddings
        for block in self.blocks:
            x = block(x, c)
        x = self.ln_f(x, c)
        logits = self.head(x)

        return logits
            