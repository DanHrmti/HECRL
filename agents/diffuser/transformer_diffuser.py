import torch
from torch import nn
from agents.diffuser.transformer_base import AdaLNEntityTransformer, SinusoidalPosEmb


class TransformerDiffuser(nn.Module):

    def __init__(self, cfg, device=None):
        super().__init__()
        projection_dim, hidden_dim, n_head, dropout = cfg.arch_eit['embed_dim'], cfg.arch_eit['h_dim'], cfg.arch_eit['n_head'], cfg.arch_eit['dropout']
        self.projection_dim = projection_dim
        self.multiview = cfg.multiview
        self.features_dim = cfg.obs_shape[-1] // 2 if cfg.obs == 'vqvae' and cfg.multiview else cfg.obs_shape[-1]
        n_layer = cfg.arch_eit['n_diffuser_layers']
        block_size = 3  # block_size is the time horizon, in our case the sequence is [obs, subgoal, goal]

        # Define an intermediate time embedding dimension.
        time_dim = projection_dim * 4

        # Time embedding network (sinusoidal position embedding).
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(projection_dim),
            nn.Linear(projection_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, projection_dim)
        )

        # Entity feature projection network.
        self.entity_projection = nn.Sequential(
            nn.Linear(self.features_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.projection_dim)
        )

        # Instantiate the AdaLN Entity Transformer.
        self.entity_transformer = AdaLNEntityTransformer(
            self.projection_dim, n_head, n_layer, block_size, self.projection_dim,
            attn_pdrop=dropout, resid_pdrop=dropout,
            hidden_dim_multiplier=4,
            activation='gelu'
        )

        # Decoder networks for entity and action outputs.
        self.entity_decoder = nn.Sequential(
            nn.Linear(self.projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.features_dim)
        )
        # Entity encoding: either shared or view-specific for multi-view inputs.
        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
        else:
            self.entity_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
        

    def forward(self, x, cond, time):
        """
        Forward pass for the denoiser.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, entity_feature_dim].
            time (torch.Tensor): Tensor of time indices with shape [batch_size]. These are embedded via time_mlp.

        Returns:
            torch.Tensor or tuple: 
                - output tensor of shape [batch_size, output_dim]
        """
        # ---------------------------------------------------------------------
        # Reshape input: separate actions and entity features.
        # x: [bs, T, entity_feature_dim]
        bs, T, f = x.size()
        x_entities = x.view(bs, T, -1, self.features_dim)
        action_entity = torch.zeros((bs, T, self.projection_dim), device=x.device)  # Placeholder for action features

        # ---------------------------------------------------------------------
        state_entities = self.entity_projection(x_entities)  # [bs, T, n_entities, projection_dim]

        if self.multiview:
            n_entities = state_entities.size(2) // 2
            entities_view1 = state_entities[:, :, :n_entities, :] + self.view1_encoding.repeat(bs, T, n_entities, 1)
            entities_view2 = state_entities[:, :, n_entities:, :] + self.view2_encoding.repeat(bs, T, n_entities, 1)
            new_state_entities = torch.cat([entities_view1, entities_view2], dim=2)
        else:
            new_state_entities = state_entities + self.entity_encoding.repeat(bs, T, state_entities.size(2), 1)

        # ---------------------------------------------------------------------
        # Prepare transformer input.

        # Time embedding: project time indices and add to all tokens.
        t_embed = self.time_mlp(time)  # [bs, projection_dim]
        x_proj = new_state_entities + t_embed[:, None, None, :]  # Broadcast addition.

        # Permute to match transformer input shape: [bs, n_tokens, T, projection_dim]
        x_proj = x_proj.permute(0, 2, 1, 3)

        # ---------------------------------------------------------------------
        # Apply the entity transformer.
        entities_trans = self.entity_transformer(x_proj, action_entity, t_embed)
        # Permute back to [bs, T, n_tokens, projection_dim].
        entities_trans = entities_trans.permute(0, 2, 1, 3)

        # ---------------------------------------------------------------------
        # Decode transformer output.
        entity_decoder_out = self.entity_decoder(entities_trans).view(bs, T, -1)  # Flatten entity outputs.
        return entity_decoder_out
