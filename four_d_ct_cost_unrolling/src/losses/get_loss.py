from .un_flow_loss import UnFlowLoss
from .constraints_loss import ConstraintsLoss
import torch.nn as nn

def get_loss(args:dict) -> nn.modules.Module:
    if "unflow" in args.loss:
        losses = {"loss_module" : UnFlowLoss(args)}

    if '2d_constraints' in args.loss:
        losses.update({"constraints_module" : ConstraintsLoss(args)})

    return losses
