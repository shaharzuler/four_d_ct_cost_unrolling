import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

class ConstraintsLoss(nn.modules.Module):
    def __init__(self, args):
        super(ConstraintsLoss, self).__init__() #TODO introduce weights!
        self.args = args

    def forward(self, pred_flow, constraints, mode='l1'): #TODO EXTRACT MODE
        constraints=constraints.to(pred_flow[0].device)
        loss = 0.0
        for flow, scale in zip(pred_flow, self.args.w_constraints_scales):
            _, _, H, W, D = flow.shape
            constraints_scaled = F.interpolate(constraints, (H, W, D), mode='area')
            ratio = H/constraints.shape[2]
            constraints_scaled *= ratio
            _,v,_ = plt.hist(constraints_scaled.abs().flatten().cpu().numpy(),bins=100)
            constraints_scaled_mask = torch.where(constraints_scaled.abs()<v[1],0,1)
            if mode == 'l1':
                loss += scale*(flow[:, :3,:,:,:]*constraints_scaled_mask - constraints_scaled).abs().mean() 
            elif mode == 'l2':
                loss += scale*((flow[:, :3,:,:,:]*constraints_scaled_mask - constraints_scaled)**2).mean() 

        return loss
