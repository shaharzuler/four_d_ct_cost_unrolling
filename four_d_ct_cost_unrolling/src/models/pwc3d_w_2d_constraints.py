from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import numpy as np

from .pwc3d import PWC3D
from .pwc_blocks import conv
from ..utils.flow_utils import rescale_flow_tensor


class PWC3Dw2dConstraints(PWC3D):
    def __init__(self, args:Dict, two_d_constraints:np.ndarray, upsample:bool=True, search_range:int=4, freeze_backbone:bool=False):
        super().__init__(args, upsample, search_range)
        if freeze_backbone:
            self.freeze_all_weights()

    def freeze_all_weights(self):
        for i in dir(self):
            if issubclass(type(eval("self."+i)), nn.Module):
                for param in eval("self."+i).parameters():
                    param.requires_grad = False
                eval("self."+i).to(torch.device('cpu'))
                
    def forward(self, data:Dict[str,torch.tensor]) -> Dict[str,torch.tensor]: 
        res_dict = super().forward(data)
        res_dict["unconstrained_flows_fw"] = res_dict["flows_fw"]
        res_dict["two_d_constraints"] = data["two_d_constraints"]
        return res_dict 

