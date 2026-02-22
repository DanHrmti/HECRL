import os
import argparse
import json
from tqdm import tqdm
from visual_encoders.dlp.models import ObjectDLP
# datasets
from visual_encoders.dlp.datasets.hecrl_ds import HECRL_Env_DS

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from piqa import PSNR, LPIPS, SSIM
except ImportError:
    print("piqa library required to compute image metrics")
    raise SystemExit


class ImageMetrics(nn.Module):
    """
    A class to calculate visual metrics between generated and ground-truth media
    """

    def __init__(self, metrics=('ssim', 'psnr', 'lpips')):
        super().__init__()
        self.metrics = metrics
        self.ssim = SSIM(reduction='none') if 'ssim' in self.metrics else None
        self.psnr = PSNR(reduction='none') if 'psnr' in self.metrics else None
        self.lpips = LPIPS(network='vgg', reduction='none') if 'lpips' in self.metrics else None

    @torch.no_grad()
    def forward(self, x, y):
        # x, y: [batch_size, 3, im_size, im_size] in [0,1]
        results = {}
        if self.ssim is not None:
            results['ssim'] = self.ssim(x, y)
        if self.psnr is not None:
            results['psnr'] = self.psnr(x, y)
        if self.lpips is not None:
            results['lpips'] = self.lpips(x, y)
        return results


def eval_im_metric(model, device, data_root_dir, val_mode='val', ds='shapes', eval_dir='./',
                   metrics=('ssim', 'psnr', 'lpips'), batch_size=32):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # load data
    if ds == "hecrl_env":
        image_size = 128
        dataset = HECRL_Env_DS(data_root_dir, mode='valid', res=image_size)
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False)

    # image metric instance
    evaluator = ImageMetrics(metrics=metrics).to(device)

    # print(len(dataloader))
    results = {}
    ssims = []
    psnrs = []
    lpipss = []
    for i, batch in enumerate(tqdm(dataloader)):
        if ds == 'shapes':
            x = batch[0].to(device)
            x_prior = x
            idx_batch = batch[1]
        elif ds == 'panda':
            x = batch[0].squeeze(1).to(device)
            x_prior = x
        else:
            raise SyntaxError(f'dataset: {ds} not recognized')
        with torch.no_grad():
            output = model(x, x_prior=x_prior, deterministic=True)
            generated = output['rec'].clamp(0, 1)
            assert x.shape[1] == generated.shape[1], "prediction and gt frames shape don't match"
            results = evaluator(x, generated)
        # [batch_size * T]
        if 'ssim' in metrics:
            ssims.append(results['ssim'])
        if 'psnr' in metrics:
            psnrs.append(results['psnr'])
        if 'lpips' in metrics:
            lpipss.append(results['lpips'])

    if 'ssim' in metrics:
        mean_ssim = torch.cat(ssims, dim=0).mean().data.cpu().item()
        results['ssim'] = mean_ssim
    if 'psnr' in metrics:
        mean_psnr = torch.cat(psnrs, dim=0).mean().data.cpu().item()
        results['psnr'] = mean_psnr
    if 'lpips' in metrics:
        mean_lpips = torch.cat(lpipss, dim=0).mean().data.cpu().item()
        results['lpips'] = mean_lpips

    # save results
    path_to_conf = os.path.join(eval_dir, 'last_val_image_metrics.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(results, outfile, indent=2)

    del evaluator  # clear memory

    return results
