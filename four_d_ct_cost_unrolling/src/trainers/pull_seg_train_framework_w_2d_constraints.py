import torch
import numpy as np
from flow_vis import flow_to_color
import cv2

from ..utils.metrics_utils import AverageMeter
from .pull_seg_train_framework import PullSegmentationMapTrainFramework
from ..utils.os_utils import torch_to_np

from ..utils.visualization_utils import disp_flow_as_arrows



class PullSegmentationMapTrainFrameworkWith2dConstraints(PullSegmentationMapTrainFramework):
    def __init__(self, train_loader, model, loss_func, args) -> None:
        super().__init__(train_loader, model, loss_func, args)
    
    def _prepare_data(self, d):
        data = super()._prepare_data(d)
        data["two_d_constraints"] = d["two_d_constraints"]
        data["two_d_constraints_with_nans"]= d["two_d_constraints_with_nans"]
        data["two_d_constraints_mask"] = torch.unsqueeze(d["two_d_constraints_mask"], 1)
        return data

    def _init_key_meters(self):
        key_meter_names = ['Loss', 'l_ph', 'l_sm', "flow_mean", "l_constraints"]
        key_meters = AverageMeter(i=len(key_meter_names), print_precision=4, names=key_meter_names)
        return key_meter_names, key_meters

    def _compute_loss_terms(self, img1, img2, vox_dim, flows, aux, data, __): 
        loss, (l_ph, l_sm, flow_mean) = super()._compute_loss_terms(img1, img2, vox_dim,flows, aux, None,None)
        l_constraints = self._get_constraints_loss(flows, data)
        loss += l_constraints

        return loss, (l_ph, l_sm, flow_mean, l_constraints)

    def _get_constraints_loss(self, flows, data):
        if "two_d_constraints" in data.keys():
            mask = ~torch.isnan(data["two_d_constraints_with_nans"])[:,:1,:,:,:]
        l_constraints = 0.0
        for loss_, module_ in self.loss_modules.items():
            if "constraints" in loss_:
                l_constraints = module_(flows, data["two_d_constraints"].to(flows[0].device), data['two_d_constraints_mask'].to(flows[0].device))
        return l_constraints

    def _add_flow_arrows_on_mask_contours_to_tensorboard(self, data, pred_flow, res_dict): 
        img1 = torch_to_np(data["template_image"][0])
        seg = torch_to_np(data["template_seg"][0])
        all_flow_arrowed_before_constraints_disp = disp_flow_as_arrows(img1, seg, torch_to_np(res_dict['unconstrained_flows_fw'][0][0][0]), text="before_constraints")
        all_flow_arrowed_constraints_disp = disp_flow_as_arrows(img1, seg, torch_to_np(res_dict['two_d_constraints'][0]), text="constraints")
        all_flow_arrowed_after_constraints_disp = disp_flow_as_arrows(img1, seg, pred_flow, text="after_constraints")
        all_flow_arrowed_disp = np.concatenate([all_flow_arrowed_before_constraints_disp, all_flow_arrowed_constraints_disp, all_flow_arrowed_after_constraints_disp], axis=2)

        self.summary_writer.add_images('sample_flows', all_flow_arrowed_disp, self.i_epoch, dataformats='NCHW')