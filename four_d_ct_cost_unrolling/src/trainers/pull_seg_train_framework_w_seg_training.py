from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from flow_n_corr_utils import disp_flow_as_arrows

from .pull_seg_train_framework import PullSegmentationMapTrainFramework
from ..utils.torch_utils import torch_to_np
from ..utils.metrics_utils import AverageMeter


class PullSegmentationMapTrainFrameworkWithSegmentation(PullSegmentationMapTrainFramework):
    def __init__(self, train_loader:Dataset, model:torch.nn.Module, loss_func:Dict[str,torch.nn.modules.Module], args:Dict) -> None:
        super().__init__(train_loader, model, loss_func, args)
    
    def _prepare_data(self, d:Dict) -> Dict:
        data = super()._prepare_data(d)
        data["template_shell_seg"] = d["template_shell_seg"]
        data["unlabeled_shell_seg"]= d["unlabeled_shell_seg"]
        return data

    def _init_key_meters(self) -> Tuple[List,AverageMeter]:
        key_meter_names = ['Loss', 'l_ph', 'l_sm', "flow_mean", "l_segmentation"]
        key_meters = AverageMeter(i=len(key_meter_names), print_precision=4, names=key_meter_names)
        return key_meter_names, key_meters

    def _compute_loss_terms(self, img1:torch.Tensor, img2:torch.Tensor, vox_dim:torch.Tensor, flows:List[torch.Tensor], aux:Tuple, data:Dict, __:Any) -> Tuple[torch.Tensor,Tuple[torch.Tensor]]: 
        loss, (l_ph, l_sm, flow_mean) = super()._compute_loss_terms(img1, img2, vox_dim, flows, aux, None, None)
        l_segmentation = self._get_segmentation_loss(flows, data)
        loss += l_segmentation

        return loss, (l_ph, l_sm, flow_mean, l_segmentation)

    def _get_segmentation_loss(self, flows:List[torch.Tensor], data:Dict[str,torch.Tensor]) -> torch.Tensor:
        l_segmentation = 0.0
        for loss_, module_ in self.loss_modules.items():
            if "segmentation" in loss_:
                l_segmentation = module_(flows, data["template_shell_seg"].to(flows[0].device), data['unlabeled_shell_seg'].to(flows[0].device))
        return l_segmentation