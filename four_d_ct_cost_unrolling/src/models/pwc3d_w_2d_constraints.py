from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import numpy as np

from .pwc3d import PWC3D
from .pwc_blocks import conv
from ..utils.flow_utils import rescale_flow_tensor


class PWC3Dw2dConstraints(PWC3D):
    def __init__(self, args:Dict, two_d_constraints:np.ndarray, upsample:bool=True, search_range:int=4, freeze_backbone:bool=True):
        super().__init__(args, upsample, search_range)
        if freeze_backbone:
            self.freeze_all_weights()
        self.two_d_constraints_network = TwoDConstraintsNetwork(two_d_constraints, num_conv_planes=args.constraints_nn_num_conv_planes)

    def freeze_all_weights(self):
        for i in dir(self):
            if issubclass(type(eval("self."+i)), nn.Module):
                for param in eval("self."+i).parameters():
                    param.requires_grad = False
                eval("self."+i).to(torch.device('cpu'))
                

    def forward(self, data:Dict[str,torch.tensor]) -> Dict[str,torch.tensor]: 
        res_dict = super().forward(data)
        constrained_flows = self.two_d_constraints_network(res_dict["flows_fw"])
        res_dict["unconstrained_flows_fw"] = res_dict["flows_fw"]
        res_dict["flows_fw"] = constrained_flows 
        res_dict["two_d_constraints"] = data["two_d_constraints"]
        return res_dict 
        
class TwoDConstraintsNetwork(nn.Module):
    def __init__(self, two_d_constraints, num_conv_planes): 
        super().__init__()
        self.convs = nn.Sequential(
            conv(in_planes=6, out_planes=num_conv_planes, kernel_size=3, stride=1, dilation=1, isReLU=True),   
            conv(in_planes=num_conv_planes, out_planes=3, kernel_size=3, stride=1, dilation=1, isReLU=False) 
        )
        self.two_d_constraints = torch.tensor(np.expand_dims(two_d_constraints, 0), requires_grad=False)
        self.device_ = None
       

    def forward(self, flows:Tuple[List[torch.Tensor],Dict[str,List]]) -> Tuple[list[torch.Tensor],Dict[str,List]]:
        if self.device_ is None:
            self.device_ = flows[0][0].device # assuming all flow pyramid levels are on the same device
            self.two_d_constraints = self.two_d_constraints.to(self.device_) 

        flows_pyramid, aux_vars = flows
        constrained_flows = []
        for flow in flows_pyramid:
            two_d_constraints_scaled = rescale_flow_tensor(self.two_d_constraints, flow.shape)
            x = torch.cat([flow, two_d_constraints_scaled], axis=1).float()
            constrained_flows.append(self.convs(x))

        return constrained_flows, aux_vars

