from .un_flow_loss import UnFlowLoss
from .l2_loss import L2Loss
from .constraints_loss import ConstraintsLoss
import torch.nn as nn

def get_loss(args:dict) -> nn.modules.Module:
    if args.loss=="unflow":
        losses = {"loss_module" : UnFlowLoss(args)}
    elif args.loss=="L2":
        losses =  {"loss_module" : L2Loss(args)}

    if '2d_constraints' in args.loss:
        losses.update({"constraints_module" : ConstraintsLoss(args)})

    return losses
