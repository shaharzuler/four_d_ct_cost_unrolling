import time
import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset
import numpy as np
import nrrd
import scipy
from scipy.ndimage.interpolation import zoom as zoom

from flow_n_corr_utils import disp_warped_img, disp_training_fig, add_mask, disp_flow_as_arrows

from .train_framework import TrainFramework
from ..utils.flow_utils import flow_warp
from ..utils.torch_utils import torch_to_np


class PullSegmentationMapTrainFramework(TrainFramework):
    def __init__(self, train_loader:Dataset, model:torch.nn.Module, loss_func:Dict[str,torch.nn.modules.Module], args:Dict) -> None:
        super().__init__(train_loader, model, loss_func, args) 
        self.reduce_loss_delay : int = 0
        self.max_reduce_loss_delay : int = args.max_reduce_loss_delay
        self.inference_args = args.inference_args

    def _run_one_epoch(self) -> bool:
        am_batch_time, am_data_time, key_meter_names, key_meters, end = self._init_epoch()
        for data in self.train_loader:
            prepared_data = self._prepare_data(data)
            am_data_time.update(time.time() - end)
            res_dict = self.model(prepared_data) 
            flows, aux = self._post_process_model_output(res_dict, data["template_image"].shape)

            loss, meters = self._compute_loss_terms(prepared_data["img1"], prepared_data["img2"], prepared_data["vox_dim"], flows, aux, prepared_data, None)
            meters = [loss, *meters]
            vals = [m.item() if torch.is_tensor(m) else m for m in meters]
            key_meters.update(vals, prepared_data["img1"].size(0))
            self._optimize(loss)
            am_batch_time.update(time.time() - end)
            end = time.time()

            self.update_to_tensorboard(key_meter_names, key_meters)
            self._visualize(data, flows[0][:,:3,:,:,:].cpu(), res_dict) 
            
            self.i_iter += 1
        avg_loss=key_meters.get_avg_meter_name("Loss")
        
        validation_data = self._create_validation_data(avg_loss, flows, data)            
        self._validate(validation_data=validation_data)

        self._update_loss_dropping(avg_loss)
        break_ = self._decide_on_early_stop()
        return break_

    def _create_validation_data(self, avg_loss, flows, data):
        validation_data = {
            "validate_self":{"avg_loss": avg_loss}, 
            "synt_validate":{
                "flows_pred": flows[0],
                "flows_gt": data["flows_gt"]
                }
            }
        if "template_seg" in data.keys():
            validation_data["synt_validate"]["template_seg"] =  data["template_seg"][0]
            validation_data["synt_validate"]["distance_validation_masks"] =  data["distance_validation_masks"]
        return validation_data

    def _visualize(self, data:Dict, pred_flow:torch.Tensor, res_dict:Dict=None) -> None: 
        self._add_orig_images_to_tensorboard(data, pred_flow)
        img1_recons_disp = self._add_warped_image_to_tensorboard(data, pred_flow)
        self._add_warped_seg_mask_to_tensorboard(data, pred_flow, img1_recons_disp)
        self._add_flow_arrows_on_mask_contours_to_tensorboard(data, torch_to_np(pred_flow[0]), res_dict)
        
    def _add_flow_arrows_on_mask_contours_to_tensorboard(self, data, pred_flow, _) -> None:
        img1 = torch_to_np(data["template_image"][0])
        seg = torch_to_np(data["template_seg"][0])
        flow_arrowed_disp = disp_flow_as_arrows(img1, seg, pred_flow, text="prediction", arrow_scale_factor=self.args.visualization_arrow_scale_factor)
        if len(data["flows_gt"].shape) > 1:
            flows_gt = torch_to_np(data["flows_gt"][0])
            gt_flow_arrowed_disp = disp_flow_as_arrows(img1, seg, flows_gt, text="ground truth", arrow_scale_factor=self.args.visualization_arrow_scale_factor)
            flow_arrowed_disp = np.concatenate([flow_arrowed_disp, gt_flow_arrowed_disp], axis=2)
        self.summary_writer.add_images('sample_flows', flow_arrowed_disp, self.i_epoch, dataformats='NCHW')

    def _add_warped_seg_mask_to_tensorboard(self, data:Dict, pred_flow:torch.Tensor, img1_recons_disp:np.ndarray) -> None:
        unlabeled_seg_map = data["unlabeled_seg"]
        seg_reconst = torch_to_np(flow_warp(unlabeled_seg_map.unsqueeze(0).float(), pred_flow, mode="nearest")).astype(bool)[0]
        warp_w_mask_disp = add_mask(img1_recons_disp[0], torch_to_np(unlabeled_seg_map)[0], seg_reconst[0])
        self.summary_writer.add_images(f'warped_seg', warp_w_mask_disp, self.i_epoch, dataformats='NHWC')

    def _add_warped_image_to_tensorboard(self, data:Dict, pred_flow:torch.Tensor) -> np.ndarray: 
        img1_recons = flow_warp(data["unlabeled_image"].unsqueeze(0), pred_flow)[0]
        img1_recons_disp = disp_warped_img(torch_to_np(data["template_image"][0]), torch_to_np(img1_recons[0]), torch_to_np(data["unlabeled_image"][0]))
        self.summary_writer.add_images(f'warped_image', img1_recons_disp, self.i_epoch, dataformats='NHWC')
        
        return img1_recons_disp

    def _add_orig_images_to_tensorboard(self, data:Dict[str,torch.Tensor], pred_flow:torch.Tensor) -> None:
        imgs_disp = disp_training_fig(torch_to_np(data["template_image"][0]), torch_to_np(data["unlabeled_image"][0]), torch_to_np(pred_flow[0]))
        self.summary_writer.add_images(f'original_images+pred_flow', imgs_disp, self.i_epoch, dataformats='NCHW')

    def infer(self, rank:int, save_mask:bool=True) -> str:
        self._init_rank(rank, update_tensorboard=False)
        self.model.eval()
        for data in self.train_loader:
            prepared_data = self._prepare_data(data)
            res_dict = self.model(prepared_data) 
            flow_tensor = res_dict["flows_fw"][0][0]
            flow_tensor = self._fix_flow_dims(flow_tensor, prepared_data["img1"].shape)
            
            if self.inference_args.inference_flow_median_filter_size:
                for axis in range(3):
                    print(f"Applying median filter on axis {axis}")
                    flow_tensor[:,axis,:,:,:] = torch.tensor(scipy.ndimage.median_filter(input=torch_to_np(flow_tensor[:,axis,:,:]), size=self.inference_args.inference_flow_median_filter_size))
            if save_mask:
                self.warp_and_save_mask(data, flow_tensor) 
        return self.output_root

    def warp_and_save_mask(self, data:Dict[str,torch.Tensor], flow:torch.Tensor, save_nrrd:bool=False) -> None: 
        template_seg_map = data["template_seg"] 
        seg_reconst = flow_warp(template_seg_map.unsqueeze(0).float(), flow.cpu(), mode="nearest")
        seg_reconst = torch_to_np(seg_reconst)[0,0,:,:,:].astype(bool)

        Path(self.inference_args.output_warped_seg_maps_dir).mkdir(parents=True, exist_ok=True)
        output_file_name = os.path.join(self.inference_args.output_warped_seg_maps_dir, f"seg_{self.inference_args.template_timestep}_to_{self.inference_args.unlabeled_timestep}")
        np.savez(f"{output_file_name}.npz", seg_reconst)
        if save_nrrd:
            nrrd.write(f"{output_file_name}.nrrd", seg_reconst.astype(int))