import torch.nn as nn
import torch

from ..utils.flow_utils import rescale_flow_tensor
from ..utils.torch_utils import rescale_mask_tensor

class ConstraintsLoss(nn.modules.Module):
    def __init__(self, args):
        super(ConstraintsLoss, self).__init__() 
        self.args = args

    def forward(self, pred_flow:list[torch.tensor], constraints:torch.tensor, mask:torch.tensor, mode:str='l1') -> torch.tensor: 
        loss = 0.0
        for flow, scale in zip(pred_flow, self.args.w_constraints_scales):
            flow = flow[:,:3,:,:,:]
            constraints_scaled, scale_factor = rescale_flow_tensor(constraints, flow.shape, return_scale_factor=True)
            mask_scaled = rescale_mask_tensor(mask, scale_factor)
            if mode == 'l1':
                loss += scale*(flow * mask_scaled - constraints_scaled).abs().mean() 
            elif mode == 'l2':
                loss += scale*((flow * mask_scaled - constraints_scaled)**2).mean() 

        return loss


