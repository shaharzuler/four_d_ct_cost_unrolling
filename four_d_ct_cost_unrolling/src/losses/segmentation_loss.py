from typing import List
import torch.nn as nn
import torch

from ..utils.flow_utils import flow_warp, rescale_flow_tensor, get_scale_factor
from ..utils.torch_utils import rescale_mask_tensor, mask_xyz_to_13xyz

class SegmentationLoss(nn.modules.Module):
    def __init__(self, args):
        super(SegmentationLoss, self).__init__() 
        self.args = args

    def forward(self, pred_flow:List[torch.Tensor], template_seg:torch.Tensor, unlabeled_seg:torch.Tensor) -> torch.Tensor: 
        loss = 0.0
        template_seg = torch.unsqueeze(template_seg, 1)
        unlabeled_seg = torch.unsqueeze(unlabeled_seg, 1)

        for flow, scale in zip(pred_flow, self.args.w_seg_scales):
            flow = flow[:,:3,:,:,:]
            scale_factor = get_scale_factor(template_seg.shape, flow.shape)
            template_seg_scaled = mask_xyz_to_13xyz(rescale_mask_tensor(template_seg, tuple(scale_factor))).float()
            unlabeled_seg_scaled = mask_xyz_to_13xyz(rescale_mask_tensor(unlabeled_seg, tuple(scale_factor))).float()
            template_seg_recons = flow_warp(unlabeled_seg_scaled.float(), flow)

            dice_coeff = (2 * (template_seg_recons * template_seg_scaled).sum()) / (template_seg_recons.sum() + template_seg_scaled.sum())
            loss += -(dice_coeff * scale)
        return loss


