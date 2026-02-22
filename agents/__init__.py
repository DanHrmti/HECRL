from agents.iql import IQL
from agents.hiql import HIQL
from agents.sgiql import SGIQL


def make_agent(cfg):
    """
	Make an agent for offline training.
	"""
    if cfg.obs in ['ec_state', 'ec_state_gen', 'dlp']:
        from agents.modules.actor_critic_eit import Actor, QNetwork, VNetwork, GoalActor
    else:
        from agents.modules.actor_critic_mlp import Actor, QNetwork, VNetwork, GoalActor

    if cfg.agent == 'iql':
        agent = IQL(cfg, Actor, QNetwork, VNetwork)
    elif cfg.agent == 'hiql':
        agent = HIQL(cfg, Actor, GoalActor, QNetwork, VNetwork)
    elif cfg.agent == 'sgiql':
        if cfg.sgiql['subgoal_policy_type'] == 'diffusion':
            from agents.diffuser.subgoal_diffuser import SubgoalDiffuser
            agent = SGIQL(cfg, Actor, SubgoalDiffuser, QNetwork, VNetwork)
        else:
            agent = SGIQL(cfg, Actor, GoalActor, QNetwork, VNetwork)
    
    return agent
