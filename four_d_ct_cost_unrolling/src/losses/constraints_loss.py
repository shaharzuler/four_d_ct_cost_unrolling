from typing import List
import torch.nn as nn
import torch

from ..utils.flow_utils import rescale_flow_tensor
from ..utils.torch_utils import rescale_mask_tensor

class ConstraintsLoss(nn.modules.Module):
    def __init__(self, args):
        super(ConstraintsLoss, self).__init__() 
        self.args = args

    def forward(self, pred_flow:List[torch.Tensor], constraints:torch.Tensor, mask:torch.Tensor, mode:str='l1') -> torch.Tensor: 
        loss = 0.0
        for flow, scale in zip(pred_flow, self.args.w_constraints_scales):
            flow = flow[:,:3,:,:,:]
            constraints_scaled, scale_factor = rescale_flow_tensor(constraints, flow.shape, return_scale_factor=True)
            mask_scaled = rescale_mask_tensor(mask, scale_factor)
            mask_scaled = mask_scaled.repeat(1,3,1,1,1) #1,1,x,y,z to 1,3,x,y,z # TODO use _mask_xyz_to_13xyz which should be moved from train_framework.py to torch_utils
            if mode == 'l1':
                loss += scale*((flow * mask_scaled) - constraints_scaled).abs().mean() #consicer mean only over nonzero
            elif mode == 'l2':
                loss += scale*(((flow * mask_scaled) - constraints_scaled)**2).mean() 

        return loss


