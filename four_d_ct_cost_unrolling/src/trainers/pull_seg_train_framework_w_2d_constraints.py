from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from flow_n_corr_utils import disp_flow_as_arrows

from .pull_seg_train_framework import PullSegmentationMapTrainFramework
from ..utils.torch_utils import torch_to_np
from ..utils.metrics_utils import AverageMeter


class PullSegmentationMapTrainFrameworkWith2dConstraints(PullSegmentationMapTrainFramework):
    def __init__(self, train_loader:Dataset, model:torch.nn.Module, loss_func:Dict[str,torch.nn.modules.Module], args:Dict) -> None:
        super().__init__(train_loader, model, loss_func, args)
    
    def _prepare_data(self, d:Dict) -> Dict:
        data = super()._prepare_data(d)
        data["two_d_constraints"] = d["two_d_constraints"]
        data["two_d_constraints_with_nans"]= d["two_d_constraints_with_nans"]
        data["two_d_constraints_mask"] = torch.unsqueeze(d["two_d_constraints_mask"], 1)
        return data

    def _init_key_meters(self) -> Tuple[List,AverageMeter]:
        key_meter_names = ['Loss', 'l_ph', 'l_sm', "flow_mean", "l_constraints"]
        key_meters = AverageMeter(i=len(key_meter_names), print_precision=4, names=key_meter_names)
        return key_meter_names, key_meters

    def _compute_loss_terms(self, img1:torch.Tensor, img2:torch.Tensor, vox_dim:torch.Tensor, flows:List[torch.Tensor], aux:Tuple, data:Dict, __:Any) -> Tuple[torch.Tensor,Tuple[torch.Tensor]]: 
        loss, (l_ph, l_sm, flow_mean) = super()._compute_loss_terms(img1, img2, vox_dim, flows, aux, None, None)
        l_constraints = self._get_constraints_loss(flows, data)
        loss += l_constraints

        return loss, (l_ph, l_sm, flow_mean, l_constraints)

    def _get_constraints_loss(self, flows:List[torch.Tensor], data:Dict[str,torch.Tensor]) -> torch.Tensor:
        l_constraints = 0.0
        for loss_, module_ in self.loss_modules.items():
            if "constraints" in loss_:
                l_constraints = module_(flows, data["two_d_constraints"].to(flows[0].device), data['two_d_constraints_mask'].to(flows[0].device))
        return l_constraints

    def _add_flow_arrows_on_mask_contours_to_tensorboard(self, data:Dict[str,torch.Tensor], pred_flow:np.ndarray, res_dict:Dict[str,torch.Tensor]) -> None: 
        img1 = torch_to_np(data["template_image"][0])
        seg = torch_to_np(data["template_LV_seg"][0])
        all_flow_arrowed_before_constraints_disp = disp_flow_as_arrows( img1, seg, torch_to_np(res_dict['unconstrained_flows_fw'][0][0][0]), text="before_constraints")
        # all_flow_arrowed_constraints_disp_sparse = disp_sparse_flow_as_arrows(img1, seg, torch_to_np(res_dict['two_d_constraints'][0]), text="constraints_as_sparse")
        all_flow_arrowed_constraints_disp_dense  = disp_flow_as_arrows( img1, seg, torch_to_np(res_dict['two_d_constraints'][0]),            text="constraints")
        all_flow_arrowed_after_constraints_disp  = disp_flow_as_arrows( img1, seg, pred_flow,                                                text="after_constraints")
        all_flow_arrowed_disp = np.concatenate([all_flow_arrowed_before_constraints_disp, all_flow_arrowed_constraints_disp_dense, all_flow_arrowed_after_constraints_disp], axis=2)
        if len(data["flows_gt"].shape) > 1:
            flows_gt = torch_to_np(data["flows_gt"][0])
            gt_flow_arrowed_disp = disp_flow_as_arrows(img1, seg, flows_gt, text="ground truth", arrow_scale_factor=self.args.visualization_arrow_scale_factor)
            all_flow_arrowed_disp = np.concatenate([all_flow_arrowed_disp, gt_flow_arrowed_disp], axis=2)
        self.summary_writer.add_images('sample_flows', all_flow_arrowed_disp, self.i_epoch, dataformats='NCHW')

    def _create_validation_data(self, avg_loss, flows, data):
        validation_data = super()._create_validation_data(avg_loss, flows, data)
        validation_data["synt_validate"]["two_d_constraints"] = data["two_d_constraints"]

        return validation_data

    def _prepare_validation_data_for_vis(self, validation_data):
        prepared_validation_data = super()._prepare_validation_data_for_vis(validation_data=validation_data)
        prepared_validation_data["two_d_constraints"] = validation_data["two_d_constraints"].to(validation_data["flows_pred"].device)
        return prepared_validation_data