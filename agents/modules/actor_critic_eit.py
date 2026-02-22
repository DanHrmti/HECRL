import torch
import torch.nn as nn

from agents.modules.base import EITblock, modulate


class Actor(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        
        self.cfg = cfg
        obs_dim, a_dim = cfg.obs_shape[-1], cfg.action_dim
        embed_dim, h_dim, n_head, dropout = cfg.arch_eit['embed_dim'], cfg.arch_eit['h_dim'], cfg.arch_eit['n_head'], cfg.arch_eit['dropout']
        out_mlp_dim = h_dim
        self.multiview = cfg.multiview
            
        if self.cfg.obs == 'dlp':
            self.projection_explicit_features = nn.Sequential(
                nn.Linear(6, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim // 2, device=device)
            )
            self.projection_visual_features = nn.Sequential(
                nn.Linear(obs_dim - 6, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim // 2, device=device)
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(obs_dim, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim, device=device)
            )

        self.self_attn1 = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='self', device=device)
        self.cross_attn = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='cross', device=device)
        self.self_attn2 = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='self', device=device)
        self.agg_attn = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='cross', device=device)

        self.out_ln = nn.LayerNorm(embed_dim, device=device)

        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.Linear(out_mlp_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.Linear(out_mlp_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.Linear(out_mlp_dim, a_dim, device=device),
            nn.Tanh()
        )

        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))

        self.out_entity = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))

    def forward(self, entities, goal_entities):

        bs, n_entities, feature_dim = entities.shape
            
        if self.cfg.obs == 'dlp':
            x = torch.cat([
                self.projection_explicit_features(entities[:, :, :6]),
                self.projection_visual_features(entities[:, :, 6:])
            ], dim=-1)
            g = torch.cat([
                self.projection_explicit_features(goal_entities[:, :, :6]),
                self.projection_visual_features(goal_entities[:, :, 6:])
            ], dim=-1)
        else:
            x = self.projection(entities)
            g = self.projection(goal_entities)

        if self.multiview:
            # add view-identifying encoding
            n_entities = x.shape[1]  # update n_entities after pre-aggregation
            x[:, :n_entities // 2] = x[:, :n_entities // 2] + self.view1_encoding.repeat(bs, n_entities // 2, 1)
            x[:, n_entities // 2:] = x[:, n_entities // 2:] + self.view2_encoding.repeat(bs, n_entities // 2, 1)
            g[:, :n_entities // 2] = g[:, :n_entities // 2] + self.view1_encoding.repeat(bs, n_entities // 2, 1)
            g[:, n_entities // 2:] = g[:, n_entities // 2:] + self.view2_encoding.repeat(bs, n_entities // 2, 1)
        
        # self-attention 1
        x = self.self_attn1(x)

        # cross-attention (goal conditioning)
        x = self.cross_attn(x, g)

        # self-attention 2
        x = self.self_attn2(x)

        # aggregation attention
        out_entity = self.out_entity.repeat(bs, 1, 1)
        x_agg = self.agg_attn(out_entity, x)

        # output mlp
        out = self.out_mlp(self.out_ln(x_agg.squeeze(1)))

        return out
	

class QNetwork(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        
        self.cfg = cfg
        obs_dim, a_dim,  = cfg.obs_shape[-1], cfg.action_dim
        embed_dim, h_dim, n_head, dropout = cfg.arch_eit['embed_dim'], cfg.arch_eit['h_dim'], cfg.arch_eit['n_head'], cfg.arch_eit['dropout']
        out_mlp_dim = h_dim
        self.multiview = cfg.multiview

        self.action_projection = nn.Sequential(
			nn.Linear(a_dim, h_dim, device=device),
            nn.GELU(),
            nn.Linear(h_dim, embed_dim, device=device)
		)
        
        if self.cfg.obs == 'dlp':
            self.projection_explicit_features = nn.Sequential(
                nn.Linear(6, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim // 2, device=device)
            )
            self.projection_visual_features = nn.Sequential(
                nn.Linear(obs_dim - 6, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim // 2, device=device)
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(obs_dim, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim, device=device)
            )
        
        self.self_attn1 = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='self', adaln=True, device=device)
        self.cross_attn = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='cross', adaln=True, device=device)
        self.self_attn2 = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='self', adaln=True, device=device)
        self.agg_attn = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='cross', adaln=True, device=device)

        self.adaln_modulation = nn.Linear(embed_dim, 2 * embed_dim, device=device)
        self.out_ln = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6, device=device)

        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.LayerNorm(out_mlp_dim, device=device),
            nn.Linear(out_mlp_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.LayerNorm(out_mlp_dim, device=device),
            nn.Linear(out_mlp_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.LayerNorm(out_mlp_dim, device=device),
            nn.Linear(out_mlp_dim, 1, device=device)
        )

        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))

        # output entity
        self.out_entity = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))

        # zero-out adaLN modulation layers
        nn.init.constant_(self.adaln_modulation.weight, 0)
        nn.init.constant_(self.adaln_modulation.bias, 0)


    def forward(self, entities, action, goal_entities):

        bs, n_entities, feature_dim = entities.shape

        # project entity features
        if self.cfg.obs == 'dlp':
            x = torch.cat([
                self.projection_explicit_features(entities[:, :, :6]),
                self.projection_visual_features(entities[:, :, 6:])
            ], dim=-1)
            g = torch.cat([
                self.projection_explicit_features(goal_entities[:, :, :6]),
                self.projection_visual_features(goal_entities[:, :, 6:])
            ], dim=-1)
        else:
            x = self.projection(entities)
            g = self.projection(goal_entities)
        
        if self.multiview:
            # add view identifying encoding
            n_entities = x.shape[1]  # update n_entities after pre-aggregation
            x[:, :n_entities // 2] = x[:, :n_entities // 2] + self.view1_encoding.repeat(bs, n_entities // 2, 1)
            x[:, n_entities // 2:] = x[:, n_entities // 2:] + self.view2_encoding.repeat(bs, n_entities // 2, 1)
            g[:, :n_entities // 2] = g[:, :n_entities // 2] + self.view1_encoding.repeat(bs, n_entities // 2, 1)
            g[:, n_entities // 2:] = g[:, n_entities // 2:] + self.view2_encoding.repeat(bs, n_entities // 2, 1)

        # project action to dimension of entities for adaLN action conditioning
        action_entity = self.action_projection(action)
        
        # self-attention 1
        x = self.self_attn1(x, adaln_c=action_entity)

        # cross-attention (goal conditioning)
        x = self.cross_attn(x, c=g, adaln_c=action_entity)

        # self-attention 2
        x = self.self_attn2(x, adaln_c=action_entity)

        # aggregation attention
        out_entity = self.out_entity.repeat(bs, 1, 1)
        x_agg = self.agg_attn(out_entity, c=x, adaln_c=action_entity)

        # output mlp
        shift, scale = self.adaln_modulation(action_entity).chunk(2, dim=-1)
        x_agg = modulate(self.out_ln(x_agg), shift, scale)
        out = self.out_mlp(x_agg.squeeze(1))

        return out
	

class VNetwork(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        
        self.cfg = cfg
        obs_dim = cfg.obs_shape[-1]
        embed_dim, h_dim, n_head, dropout = cfg.arch_eit['embed_dim'], cfg.arch_eit['h_dim'], cfg.arch_eit['n_head'], cfg.arch_eit['dropout']
        out_mlp_dim = h_dim
        self.multiview = cfg.multiview

        if self.cfg.obs == 'dlp':
            self.projection_explicit_features = nn.Sequential(
                nn.Linear(6, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim // 2, device=device)
            )
            self.projection_visual_features = nn.Sequential(
                nn.Linear(obs_dim - 6, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim // 2, device=device)
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(obs_dim, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim, device=device)
            )
        
        self.self_attn1 = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='self', device=device)
        self.cross_attn = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='cross', device=device)
        self.self_attn2 = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='self', device=device)
        self.agg_attn = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='cross', device=device)

        self.out_ln = nn.LayerNorm(embed_dim, device=device)

        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.Linear(out_mlp_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.Linear(out_mlp_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.Linear(out_mlp_dim, 1, device=device)
        )

        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))
        
        self.out_entity = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))

    def forward(self, entities, goal_entities):

        bs, n_entities, feature_dim = entities.shape

        # project entity features
        if self.cfg.obs == 'dlp':
            x = torch.cat([
                self.projection_explicit_features(entities[:, :, :6]),
                self.projection_visual_features(entities[:, :, 6:])
            ], dim=-1)
            g = torch.cat([
                self.projection_explicit_features(goal_entities[:, :, :6]),
                self.projection_visual_features(goal_entities[:, :, 6:])
            ], dim=-1)
        else:
            x = self.projection(entities)
            g = self.projection(goal_entities)

        if self.multiview:
            # add view identifying encoding
            n_entities = x.shape[1]  # update n_entities after pre-aggregation
            x[:, :n_entities // 2] = x[:, :n_entities // 2] + self.view1_encoding.repeat(bs, n_entities // 2, 1)
            x[:, n_entities // 2:] = x[:, n_entities // 2:] + self.view2_encoding.repeat(bs, n_entities // 2, 1)
            g[:, :n_entities // 2] = g[:, :n_entities // 2] + self.view1_encoding.repeat(bs, n_entities // 2, 1)
            g[:, n_entities // 2:] = g[:, n_entities // 2:] + self.view2_encoding.repeat(bs, n_entities // 2, 1)
    
        # self-attention 1
        x = self.self_attn1(x)

        # cross-attention (goal conditioning)
        x = self.cross_attn(x, g)

        # self-attention 2
        x = self.self_attn2(x)
        
        # aggregation attention
        out_entity = self.out_entity.repeat(bs, 1, 1)
        x_agg = self.agg_attn(out_entity, x)

        # output mlp
        out = self.out_mlp(self.out_ln(x_agg.squeeze(1)))

        return out


class GoalActor(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        
        self.cfg = cfg
        obs_dim, a_dim = cfg.obs_shape[-1], cfg.action_dim
        embed_dim, h_dim, n_head, dropout = cfg.arch_eit['embed_dim'], cfg.arch_eit['h_dim'], cfg.arch_eit['n_head'], cfg.arch_eit['dropout']
        out_mlp_dim = h_dim
        self.multiview = cfg.multiview
            
        if self.cfg.obs == 'dlp':
            self.projection_explicit_features = nn.Sequential(
                nn.Linear(6, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim // 2, device=device)
            )
            self.projection_visual_features = nn.Sequential(
                nn.Linear(obs_dim - 6, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim // 2, device=device)
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(obs_dim, h_dim, device=device),
                nn.LayerNorm(h_dim, device=device),
                nn.GELU(),
                nn.Linear(h_dim, embed_dim, device=device)
            )

        self.self_attn1 = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='self', device=device)
        self.cross_attn = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='cross', device=device)
        self.self_attn2 = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='self', device=device)
        self.self_attn3 = EITblock(embed_dim, h_dim, n_head, dropout=dropout, attn_type='self', device=device)

        self.out_ln = nn.LayerNorm(embed_dim, device=device)

        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.Linear(out_mlp_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.Linear(out_mlp_dim, out_mlp_dim, device=device),
            nn.GELU(),
            nn.Linear(out_mlp_dim, obs_dim, device=device),
        )

        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim, device=device))

    def forward(self, entities, goal_entities):

        bs, n_entities, feature_dim = entities.shape

        # project entity features
        if self.cfg.obs == 'dlp':
            x = torch.cat([
                self.projection_explicit_features(entities[:, :, :6]),
                self.projection_visual_features(entities[:, :, 6:])
            ], dim=-1)
            g = torch.cat([
                self.projection_explicit_features(goal_entities[:, :, :6]),
                self.projection_visual_features(goal_entities[:, :, 6:])
            ], dim=-1)
        else:
            x = self.projection(entities)
            g = self.projection(goal_entities)

        if self.multiview:
            # add view identifying encoding
            n_entities = x.shape[1]  # update n_entities after pre-aggregation
            x[:, :n_entities // 2] = x[:, :n_entities // 2] + self.view1_encoding.repeat(bs, n_entities // 2, 1)
            x[:, n_entities // 2:] = x[:, n_entities // 2:] + self.view2_encoding.repeat(bs, n_entities // 2, 1)
            g[:, :n_entities // 2] = g[:, :n_entities // 2] + self.view1_encoding.repeat(bs, n_entities // 2, 1)
            g[:, n_entities // 2:] = g[:, n_entities // 2:] + self.view2_encoding.repeat(bs, n_entities // 2, 1)
        
        # self-attention 1
        x = self.self_attn1(x)

        # cross-attention (goal conditioning)
        x = self.cross_attn(x, g)

        # self-attention 2
        x = self.self_attn2(x)

        # self-attention 3
        x = self.self_attn3(x)

        # output mlp
        out = self.out_mlp(self.out_ln(x))

        return out
    