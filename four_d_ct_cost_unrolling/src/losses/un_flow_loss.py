import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_loss import TernaryLoss, smooth_grad_1st, SSIM
from ..utils.flow_utils import flow_warp


class UnFlowLoss(nn.modules.Module):
    def __init__(self, args):
        super(UnFlowLoss, self).__init__()
        self.args = args
    
    def loss_admm(self, q, c, beta):
        loss = []

        if self.args.w_admm > 0:
            loss += [(q - c + beta)**2]

        return self.args.w_admm * self.args.admm_rho / 2 * sum([l.mean() for l in loss])

    def loss_photometric(self, img1_scaled:torch.tensor, img1_recons:torch.tensor) -> torch.tensor:
        loss = []

        if self.args.w_l1 > 0:
            loss += [self.args.w_l1 * (img1_scaled - img1_recons).abs()]

        elif self.args.w_l2 > 0:
            loss += [self.args.w_l2 * ((img1_scaled - img1_recons)**2)]

        if self.args.w_ssim > 0:
            loss += [self.args.w_ssim * SSIM(img1_recons, img1_scaled)]

        if self.args.w_ternary > 2:
            loss += [self.args.w_ternary * TernaryLoss(img1_recons, img1_scaled)]

        return sum([l.mean() for l in loss])

    def loss_smooth(self, flow:torch.tensor, vox_dim:torch.tensor) -> torch.tensor:
        func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, vox_dim)]
        return sum([l.mean() for l in loss])

    def forward(self, output:list, img1:torch.tensor, img2:torch.tensor, aux:tuple, vox_dim:torch.tensor) -> tuple[torch.tensor]:
        vox_dim = vox_dim.squeeze(0)

        pyramid_flows = output

        pyramid_warp_losses = []
        pyramid_smooth_losses = []

        for i, flow in enumerate(pyramid_flows):
            N, C, H, W, D = flow.size()
            img1_scaled = F.interpolate(img1, (H, W, D), mode='area')
            img2_scaled = F.interpolate(img2, (H, W, D), mode='area')
            flow12 = flow[:, :3]
            img1_recons = flow_warp(img2_scaled, flow12)

            loss_smooth = self.loss_smooth(flow=flow12, vox_dim=vox_dim)
            loss_photometric = self.loss_photometric(img1_scaled, img1_recons)
            pyramid_smooth_losses.append(loss_smooth)
            pyramid_warp_losses.append(loss_photometric)


        pyramid_warp_losses = [l * w for l, w in zip(pyramid_warp_losses, self.args.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in zip(pyramid_smooth_losses, self.args.w_sm_scales)]

        loss_smooth = sum(pyramid_smooth_losses)
        loss_photometric = sum(pyramid_warp_losses)

        loss_total = loss_smooth + loss_photometric

        return loss_total, loss_photometric, loss_smooth, pyramid_flows[0].abs().mean()

