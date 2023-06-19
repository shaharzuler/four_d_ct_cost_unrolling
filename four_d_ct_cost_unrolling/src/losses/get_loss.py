from .un_flow_loss import UnFlowLoss
from .l2_loss import L2Loss
from .constraints_loss import ConstraintsLoss

def get_loss(loss:str, args=None):
    if loss=="L1":
        losses = {"loss_module" : UnFlowLoss(args)}
    elif loss=="L2":
        losses =  {"loss_module" : L2Loss(args)}

    if '2d_constraints' in loss:
        losses.update({"constraints_module" : ConstraintsLoss(args)})

    return losses
