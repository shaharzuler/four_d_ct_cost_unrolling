import time
from typing import Any, Dict, List, Tuple

from scipy.ndimage.interpolation import zoom as zoom
import torch
from torch.utils.data import Dataset
import numpy as np

import three_d_data_manager
from flow_n_corr_utils import disp_flow_error_colors

from .base_trainer import BaseTrainer
from ..utils.flow_utils import flow_warp
from ..utils.metrics_utils import AverageMeter, calc_epe_error, calc_error_in_mask, calc_error_on_surface, calc_error_vs_distance, get_error_vs_distance_plot_image
from ..utils.torch_utils import torch_to_np



class TrainFramework(BaseTrainer):
    def __init__(self, train_loader:Dataset, model:torch.nn.Module, loss_func:Dict[str,torch.nn.modules.Module], args:Dict):
        super(TrainFramework, self).__init__(train_loader, model, loss_func, args)

    def _compute_loss_terms(self, img1:torch.Tensor, img2:torch.Tensor, vox_dim:torch.Tensor, flows:List[torch.Tensor], aux:Tuple, _:Any, __:Any) -> Tuple[torch.Tensor,Tuple[torch.Tensor]]: 
        loss, l_ph, l_sm, flow_mean = self.loss_modules['loss_module'](flows, img1, img2, aux, vox_dim)
        return loss, (l_ph, l_sm, flow_mean)

    def _fix_flow_dims(self, flow, shape_):
         if (flow.shape)[-3:] != shape_[-3:]:
            diff = np.array(flow.shape[-3:])-shape_[-3:]
            diff_a = diff//2
            diff_b = diff-diff_a
            
            return flow[:,:,diff_a[0]:-diff_b[0] or None, diff_a[1]:-diff_b[1] or None,diff_a[2]:-diff_b[2] or None] #TODO generalize it!


    def _post_process_model_output(self, res_dict:Dict[str,torch.Tensor], shape_:Tuple) -> Tuple[List[torch.Tensor],Tuple]:
        flows = res_dict['flows_fw'][0]
        flows[0] = self._fix_flow_dims(flows[0], shape_)

        aux = res_dict['flows_fw'][1]
        return flows, aux

    def update_to_tensorboard(self, key_meter_names:List[str], key_meters:AverageMeter) -> None:
        if self.i_iter % self.args.record_freq == 0:
            for v, name in zip(key_meters.val, key_meter_names):
                self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)

    def _optimize(self, loss:torch.Tensor) -> None:
        loss = loss.mean()
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
    def _init_epoch(self) -> Tuple[AverageMeter,AverageMeter,List,AverageMeter,float]:
        avg_meter_batch_time = AverageMeter()
        avg_meter_data_time = AverageMeter()
        self.model.train()

        key_meter_names, key_meters = self._init_key_meters()
        end = time.time()
        return avg_meter_batch_time, avg_meter_data_time, key_meter_names, key_meters, end

    def _init_key_meters(self) -> Tuple[List, AverageMeter]:
        key_meter_names = ['Loss', 'l_ph', 'l_sm', "flow_mean"]
        key_meters = AverageMeter(i=len(key_meter_names), print_precision=4, names=key_meter_names)
        return key_meter_names, key_meters

    def _prepare_data(self, data:Dict) -> Dict:
        img1, img2 = [im.unsqueeze(0).float().to(self.rank) for im in [data["template_image"], data["unlabeled_image"]]]
        vox_dim = torch.tensor([[1,1,1.]], dtype=torch.float64).to(self.rank)
        
        data = { 
            "img1": img1,
            "img2": img2,
            "vox_dim": vox_dim,
        }

        return data 
            
    @torch.no_grad()
    def _validate(self, validation_data:Dict) -> None:
        if hasattr(self.args,'dump_disp') and self.args.dump_disp:
            return self._dumpt_disp_fields()
        else:
            if 'synthetic' in self.args.valid_type:
                self._synt_validate(validation_data["synt_validate"])
            if 'variance_valid' in self.args.valid_type:
                self.variance_validate()
            if "basic" in self.args.valid_type: 
                self._validate_self(validation_data["validate_self"])

    def _validate_basic(self, validate_data:Dict) -> None: # optional - also validate by iou
        if self.i_iter > self.args.save_iter:
            self._save_model(validate_data["avg_loss"], name=self.model_suffix) 

    def _validate_self(self, validate_self_data:Dict) -> None:
        self._validate_basic(validate_self_data) 

    def _synt_validate(self, validation_data): 
        prepared_validation_data = self._prepare_validation_data_for_vis(validation_data)
        self._compute_and_plot_validation_errors(validation_data, **prepared_validation_data)
        self.add_flow_error_vis_to_tensorboard(**prepared_validation_data)

    def _prepare_validation_data_for_vis(self, validation_data):
        flows_pred = validation_data["flows_pred"]
        flows_gt = validation_data["flows_gt"].to(flows_pred.device)
        processed_validation_data = {"flows_pred":flows_pred, "flows_gt":flows_gt}
        if 'distance_validation_masks' in validation_data.keys():
            processed_validation_data["distance_validation_masks"] = {}
            for region_name, region in validation_data["distance_validation_masks"].items():
                processed_validation_data["distance_validation_masks"][region_name] = {}
                for distance, distance_mask in region.items():
                    processed_validation_data["distance_validation_masks"][region_name][distance] = distance_mask[0]

        return processed_validation_data
        

    def _compute_and_plot_validation_errors(self, validation_data, flows_pred, flows_gt, distance_validation_masks, **qwargs):
        complete_error = calc_epe_error(flows_gt, flows_pred)
        self.summary_writer.add_scalar('Validation Error',complete_error,self.i_epoch)
        
        volume_error = calc_error_in_mask(flows_gt, flows_pred, validation_data["template_seg"])
        self.summary_writer.add_scalar('Validation LV Volume Error', volume_error, self.i_epoch)

        surface_error = calc_error_on_surface(flows_gt, flows_pred, validation_data["template_seg"])
        self.summary_writer.add_scalar('Validation Surface Error', surface_error, self.i_epoch)

        distance_calculated_errors, rel_distance_calculated_errors = calc_error_vs_distance(flows_pred, flows_gt, distance_validation_masks)

        for region_name, region in distance_calculated_errors.items():
            for distance, distance_error in zip(*region):
                self.summary_writer.add_scalar(f'Distance {region_name} Validation Error/{distance}', np.array(distance_error), self.i_epoch)
        
        error_vs_dist_plot = get_error_vs_distance_plot_image(distance_validation_masks, distance_calculated_errors)   
        self.summary_writer.add_images(f'Distance Validation Error', error_vs_dist_plot, self.i_epoch, dataformats='NHWC')
        
        for region_name, region in rel_distance_calculated_errors.items():
            for distance, distance_error in zip(*region):
                self.summary_writer.add_scalar(f'Distance {region_name} Relative Validation Error/{distance}', np.array(distance_error), self.i_epoch)
        
        rel_error_vs_dist_plot = get_error_vs_distance_plot_image(distance_validation_masks, rel_distance_calculated_errors)   
        self.summary_writer.add_images(f'Relative Distance Validation Error', rel_error_vs_dist_plot, self.i_epoch, dataformats='NHWC')

    def add_flow_error_vis_to_tensorboard(self, flows_pred:torch.Tensor, flows_gt:torch.Tensor, two_d_constraints:torch.Tensor=None, **qwargs) -> None: 
        flow_colors_error_disp = disp_flow_error_colors(torch_to_np(flows_pred[0]), torch_to_np(flows_gt[0]), torch_to_np(two_d_constraints[0]) if two_d_constraints is not None else None)
        self.summary_writer.add_images(f'flow_error', flow_colors_error_disp, self.i_epoch, dataformats='NCHW')


    @torch.no_grad()
    def variance_validate(self): # NOT TESTED
        error_median = 0
        error_mean = 0
        error_short = 0
        max_diff_error = 0
        frame_diff_error = 0
        error_median_box = 0
        error_mean_box = 0
        error_short_box = 0
        max_diff_error_box = 0
        frame_diff_error_box = 0
        loss = 0
        im_h = im_w = 192
        im_d = 64
        flows = torch.zeros([3, im_h, im_w, im_d], device=self.device)
        images_warped = torch.zeros(
            [self.args.variance_valid_len, im_h, im_w, im_d], device=self.device)

        for i_step, data in enumerate(self.valid_loader):

            # Prepare data
            img1, img2, name = data
            vox_dim = img1[1].to(self.device)
            img1, img2 = img1[0].to(self.device), img2[0].to(self.device)
            img1 = img1.unsqueeze(1).float()  # Add channel dimension
            img2 = img2.unsqueeze(1).float()  # Add channel dimension

            if i_step % (self.args.variance_valid_len - 1) == 0:
                image0 = img1
                images_warped[i_step %
                              (self.args.variance_valid_len - 1)] = img1.squeeze(0)
                count = 0

            # Remove batch dimension, net prediction
            res = self.model(img1, img2, vox_dim=vox_dim)['flows_fw'][0][0].squeeze(0).float()
            flows += res
            images_warped[(i_step % (self.args.variance_valid_len - 1))+1] = flow_warp(img2,flows.unsqueeze(0))  # im1 recons
            count += 1

            if count == self.args.variance_valid_short_len - 1:
                variance = torch.std(images_warped[:count + 1, :, :, :], dim=0)
                error_short += float(variance.mean().item())
                box_variance = variance[49:148, 49:148, 16:48]
                error_short_box += float(box_variance.mean().item())

            if count == self.args.frame_dif+1:
                # calculating variance based only on model
                res = self.model(image0, img2, vox_dim=vox_dim)['flows_fw'][0][0].squeeze(0).float()
                diff_warp_straight = torch.zeros([2, im_h, im_w, im_d], device=self.device)
                diff_warp_straight[0] = images_warped[0]
                diff_warp_straight[1] = flow_warp(img2, res.unsqueeze(0))
                diff_variance_straight = torch.std(diff_warp_straight, dim=0)
                frame_diff_error += float(diff_variance_straight.median().item())
                box_variance = diff_variance_straight[49:148, 49:148, 16:48]
                frame_diff_error_box += float(box_variance.mean().item())
            if count == self.args.variance_valid_len - 1:
                # calculating max_diff variance
                diff_warp = torch.zeros([2, im_h, im_w, im_d], device=self.device)
                diff_warp[0] = images_warped[0]
                diff_warp[1] = images_warped[-1]
                diff_variance = torch.std(diff_warp, dim=0)
                max_diff_error += float(diff_variance.mean().item())
                box_variance = diff_variance[49:148, 49:148, 16:48]
                max_diff_error_box += float(box_variance.mean().item())
                
                variance = torch.std(images_warped, dim=0)
                error_median += float(variance.median().item())
                error_mean += float(variance.mean().item())
                box_variance = variance[49:148, 49:148, 16:48]
                error_mean_box += float(box_variance.mean().item())
                error_median_box += float(box_variance.median().item())
                flows = torch.zeros([3, im_h, im_w, im_d], device=self.device)
                count = 0

        max_diff_error /= self.args.variance_valid_sets
        frame_diff_error /= self.args.variance_valid_sets
        error_median /= self.args.variance_valid_sets
        error_mean /= self.args.variance_valid_sets
        error_short /= self.args.variance_valid_sets

        max_diff_error_box /= self.args.variance_valid_sets
        frame_diff_error_box /= self.args.variance_valid_sets
        error_median_box /= self.args.variance_valid_sets
        error_mean_box /= self.args.variance_valid_sets
        error_short_box /= self.args.variance_valid_sets
        print(f'Validation maxDiff error-> {max_diff_error}, Validation error mean -> {error_mean}, Validation error median -> {error_median} Short Validation error -> {error_short}')

        self.writer.add_scalar('Validation Difference_Error', max_diff_error, self.i_epoch)
        self.writer.add_scalar('Validation frame_difference_Error',frame_diff_error, self.i_epoch)
        self.writer.add_scalar('Validation Error(mean)', error_mean,self.i_epoch)
        self.writer.add_scalar('Validation Error(median)',error_median,self.i_epoch)
        self.writer.add_scalar('Validation Short Error',error_short,self.i_epoch)
        self.writer.add_scalar('Validation Difference_Error_box',max_diff_error_box,self.i_epoch)
        self.writer.add_scalar('Validation frame_difference_Error_box',frame_diff_error_box,self.i_epoch)
        self.writer.add_scalar('Validation Error(mean)_box',error_mean_box,self.i_epoch)
        self.writer.add_scalar('Validation Error(median)_box',error_median_box,self.i_epoch)
        self.writer.add_scalar('Validation Short Error_box',error_short_box,self.i_epoch)
 
        p_valid = plot_images(images_warped[0].detach().cpu(), images_warped[-1].detach().cpu(), img2.detach().cpu(), show=False)
        
        self.writer.add_figure('Valid_Images_original', p_valid, self.i_epoch)
        p_dif_valid = plot_images(images_warped[0].detach().cpu(), diff_warp[-1].detach().cpu(), img2.detach().cpu(), show=False)
        p_dif_col = plot_warped_img(images_warped[0].detach().cpu(), images_warped[-1].detach().cpu())
        self.writer.add_figure('Valid_Images_warped', p_dif_col, self.i_epoch)

        return [error_median], ["error_median"]

