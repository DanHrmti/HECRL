import torch
import torch.nn.functional as F


def batch_pairwise_dist(x, y, metric='l2_simple'):
	assert metric in ['l2', 'l2_simple', 'l1', 'cosine'], f'metric {metric} unrecognized'
	bs, num_points_x, points_dim = x.size()
	_, num_points_y, _ = y.size()
	if metric == 'cosine':
		dist_func = torch.nn.functional.cosine_similarity
		P = -dist_func(x.unsqueeze(2), y.unsqueeze(1), dim=-1, eps=1e-8)
	elif metric == 'l1':
		P = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(-1)
	elif metric == 'l2_simple':
		P = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
	else:
		xx = torch.bmm(x, x.transpose(2, 1))
		yy = torch.bmm(y, y.transpose(2, 1))
		zz = torch.bmm(x, y.transpose(2, 1))
		diag_ind_x = torch.arange(0, num_points_x, device=x.device)
		diag_ind_y = torch.arange(0, num_points_y, device=y.device)
		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = rx.transpose(2, 1) + ry - 2 * zz
	return P


def chamfer_loss(pred, target, cfg):
	
	bs, n_entities, feature_dim = pred.shape
	if cfg.multiview:
		# calculate chamfer distance for each view seperately
		pred = pred.reshape(bs * 2, n_entities // 2, feature_dim)
		target = target.reshape(bs * 2, n_entities // 2, feature_dim)

	P = batch_pairwise_dist(pred, target, cfg.chamfer_metric)

	# compute dist from target to pred
	entity_dist1, min_indices = torch.min(P, 1)

	# compute dist from pred to target
	entity_dist2, min_indices = torch.min(P, 2)
	
	# compute loss
	# NOTE: give more weight to dist from target to generated (entity_dist1) since target is ground truth and should be an anchor
	chamfer_dist = (cfg.chamfer_target_weight * entity_dist1 + entity_dist2) / (cfg.chamfer_target_weight + 1)
	if cfg.multiview:
		chamfer_dist = chamfer_dist.reshape(bs, n_entities)
	chamfer_loss = chamfer_dist.mean(dim=-1, keepdim=True)
	return chamfer_loss
