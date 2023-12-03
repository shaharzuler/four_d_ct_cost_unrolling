from typing import Dict
import torch.nn as nn

from .un_flow_loss import UnFlowLoss
from .constraints_loss import ConstraintsLoss
from .segmentation_loss import SegmentationLoss

def get_loss(args:Dict) -> Dict[str,nn.modules.Module]:
    if "unflow" in args.loss:
        losses = {"loss_module" : UnFlowLoss(args)}

    if '2d_constraints' in args.loss:
        losses.update({"constraints_module" : ConstraintsLoss(args)})
    
    if 'segmentation' in args.loss:
        losses.update({"segmentation_module" : SegmentationLoss(args)})

    return losses
