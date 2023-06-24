from .pwc3d import PWC3D
from .pwc_blocks import conv
import torch
import torch.nn as nn
import torch.nn.functional as F




class PWC3Dw2dConstraints(PWC3D):
    def __init__(self, args, upsample=True, reduce_dense=True, search_range=4, freeze_backbone=True):
        super().__init__(args, upsample, reduce_dense, search_range)
        if freeze_backbone:
            self.freeze_all_weights()
        self.two_d_constraints_network = TwoDConstraintsNetwork().cuda() # TODO make cunfigurable

    def freeze_all_weights(self):
        for i in dir(self):
            if issubclass(type(eval("self."+i)), nn.Module):
                for param in eval("self."+i).parameters():
                    param.requires_grad = False
                eval("self."+i).to(torch.device('cpu'))
                

    def forward(self, data, w_bk=True): 
        res_dict = super().forward(data, w_bk)
        constrained_flows = self.two_d_constraints_network(res_dict["flows_fw"], data["2d_constraints"])
        # TODO handle backwards (requires backwards constraints!). 
        res_dict["unconstrained_flows_fw"] = res_dict["flows_fw"]
        res_dict["flows_fw"] = constrained_flows 
        res_dict["2d_constraints"] = data["2d_constraints"]
        return res_dict 
        
class TwoDConstraintsNetwork(nn.Module):
    def __init__(self, num_neurons=36): #num_neurons=48): 
        super().__init__()
        self.convs = nn.Sequential(
            conv(in_planes=6, out_planes=num_neurons, kernel_size=3, stride=1, dilation=1, isReLU=True),   
            conv(in_planes=num_neurons, out_planes=3, kernel_size=3, stride=1, dilation=1, isReLU=False) 
        )
       

    def forward(self, flows, two_d_constraints):
        flows_pyramid, aux_vars = flows
        constrained_flows = []
        old_two_d_constraints_device = two_d_constraints.device
        two_d_constraints=two_d_constraints.to(flows_pyramid[0].device) # assuming all flow pyramid levels are on the same device
        for flow in flows_pyramid:
            ratio = flow.shape[-1] / two_d_constraints.shape[-1]
            if ratio != 1.:
                two_d_constraints_scaled = F.interpolate(two_d_constraints * ratio, scale_factor=ratio, mode='trilinear')
            else:
                two_d_constraints_scaled = two_d_constraints
            x = torch.cat([flow, two_d_constraints_scaled], axis=1).float()
            constrained_flows.append(self.convs(x))

        old_two_d_constraints_device = two_d_constraints.to(old_two_d_constraints_device)
        return constrained_flows, aux_vars

