from .train_framework import TrainFramework
from ..utils.visualization_utils import disp_warped_img, disp_training_fig, add_mask, disp_flow_as_arrows
from ..utils.flow_utils import flow_warp
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
import torch
import time
import scipy
import re
import nrrd
import os
import cv2

from ..utils.os_utils import torch_to_np



class PullSegmentationMapTrainFramework(TrainFramework):
    def __init__(self, train_loader, model, loss_func, args) -> None:
        super().__init__(train_loader, model, loss_func, args)
        self.reduce_loss_delay : int = 0
        self.old_avg_loss : float = 1E10
        self.max_reduce_loss_delay : int = args.max_reduce_loss_delay

    def _run_one_epoch(self) -> bool:
        am_batch_time, am_data_time, key_meter_names, key_meters, end = self._init_epoch()

        for data in self.train_loader:

            prepared_data = self._prepare_data(data)
            am_data_time.update(time.time() - end)
            res_dict = self.model(prepared_data) 
            flows, aux = self._post_process_model_output(res_dict)

            loss, meters = self._compute_loss_terms(prepared_data["img1"], prepared_data["img2"], prepared_data["vox_dim"], flows, aux, None, None)
            meters = [loss, *meters]
            vals = [m.item() if torch.is_tensor(m) else m for m in meters]
            key_meters.update(vals, prepared_data["img1"].size(0))
            self._optimize(loss)
            am_batch_time.update(time.time() - end)
            end = time.time()

            self.update_to_tensorboard(key_meter_names, key_meters)
            self._visualize(data, flows[0][:,:3,:,:,:].cpu()) 
            
            self.i_iter += 1
        avg_loss=key_meters.get_avg_meter_name("Loss")
        validation_data = {
            "validate_self":{"avg_loss": avg_loss}, 
            "synt_validate":{
                "flows_pred": flows,
                "flows_gt": data["flows_gt"]
                }
            }
        self._validate(validation_data=validation_data)
        
        if avg_loss < self.old_avg_loss:
            self.old_avg_loss = avg_loss
            self.reduce_loss_delay = 0
        else:
            self.reduce_loss_delay += 1
            
        if self.reduce_loss_delay > self.max_reduce_loss_delay:
            break_ = True
            return break_
        else:
            break_ = False
            return break_


    def _visualize(self, data:dict, pred_flow:torch.tensor):   # TODO REFACTOR THIS MESS # img1->template_image, img2->unlabeled_image 
        self._add_orig_images_to_tensorboard(data, pred_flow)
        img1_recons_disp = self._add_warped_image_to_tensorboard(data, pred_flow)
        self._add_warped_seg_mask_to_tensorboard(data, pred_flow, img1_recons_disp)
        self._add_flow_arrows_on_mask_contours_to_tensorboard(data, torch_to_np(pred_flow[0]))

    def _add_flow_arrows_on_mask_contours_to_tensorboard(self, data, pred_flow):
        img1 = torch_to_np(data["template_image"][0])
        seg = torch_to_np(data["template_seg"][0])
        all_flow_arrowed_disp = disp_flow_as_arrows(img1, seg, pred_flow)
        self.summary_writer.add_images('sample_flows', all_flow_arrowed_disp, self.i_epoch, dataformats='NCHW')

    def _add_warped_seg_mask_to_tensorboard(self, data:dict, pred_flow:torch.tensor, img1_recons_disp:np.array) -> None:
        template_seg_map = data["template_seg"]
        seg_reconst = torch_to_np(flow_warp(template_seg_map.unsqueeze(0).float(), pred_flow, mode="nearest")).astype(bool)[0]
        warp_w_mask_disp = add_mask(img1_recons_disp[0], torch_to_np(template_seg_map)[0], seg_reconst[0])
        self.summary_writer.add_images(f'warped_seg', warp_w_mask_disp, self.i_epoch, dataformats='NHWC')


    def _add_warped_image_to_tensorboard(self, data:dict, pred_flow:torch.tensor) -> np.array: #TODO validate with 2 different images
        img1_recons = flow_warp(data["unlabeled_image"].unsqueeze(0), pred_flow)[0]
        img1_recons_disp = disp_warped_img(torch_to_np(data["unlabeled_image"][0]), torch_to_np(img1_recons[0]), torch_to_np(data["template_image"][0]))
        self.summary_writer.add_images(f'warped_image', img1_recons_disp, self.i_epoch, dataformats='NHWC')
        
        return img1_recons_disp

    def _add_orig_images_to_tensorboard(self, data, pred_flow):
        imgs_disp = disp_training_fig(torch_to_np(data["template_image"][0]), torch_to_np(data["unlabeled_image"][0]), torch_to_np(pred_flow[0]))
        self.summary_writer.add_images(f'original_images', imgs_disp, self.i_epoch, dataformats='NCHW')



class PullSegmentationMapTrainFrameworkInference(PullSegmentationMapTrainFramework):
    def __init__(self, train_loader, valid_loader, model, loss_func, args) -> None:
        super().__init__(train_loader, valid_loader, model, loss_func, args)
        self.timestep_num = self.extract_timestep_num_from_ckpts_name(args['load'])
        self.warped_seg_maps_dir = "warped_seg_maps"
        self.flow_median_filter = True


    def _validate_batch(self, batch_time, end, i_step, data):
        for d in data:
            prepared_data = self._prepare_data(d) # img1.shape == img2.shape is now [1, 1, 192, 192, 192]
            flows = self.model(prepared_data, w_bk=False)['flows_fw'][0][0]
            pred_flows = flows.detach().squeeze(0)

            if self.flow_median_filter:
                for axis in range(pred_flows.shape[0]):
                    print(f"Applying median filter on axis {axis}")
                    pred_flows[axis,:,:,:] = torch.tensor(torch_to_np(scipy.ndimage.median_filter(input=pred_flows[axis,:,:]), size=7))
                    
            self.save_warped_mask(d, pred_flows) 

    def save_warped_mask(self, data, pred_flows, save_nrrd=False): #TODO REFACTOR COMPLETELY
        template_seg_map = data["template_seg_map"] # template_seg_map.shape is now 1, 192, 192, 192
        seg_reconst = flow_warp(template_seg_map.unsqueeze(0).float(), pred_flows.cpu().unsqueeze(0)) # seg_reconst.shape is now 1, 1, 192, 192, 192. floats!
        # TODO convert seg_reconst to int
        seg = torch_to_np(seg_reconst)[0,0,:,:,:]

        img3d = np.rot90(seg,k=2,axes=(0,1))
        img3d = np.rot90(img3d,k=2,axes=(1,2)) 
        img3d = np.rot90(img3d,k=1)

        seg_zoomed = scipy.ndimage.zoom(img3d.astype(bool).astype(int),zoom=(256/192,256/192,91/192),order=0, mode="nearest")#2)  shape is now 256,256,91
        seg_final = np.moveaxis(seg_zoomed, -1, 0)
        # seg_final = (seg_final>0.1).astype(int) # "booliazation" and cast to int

        np.savez(os.path.join(self.warped_seg_maps_dir, f"seg_{self.valid_set.patient_index}_{self.valid_set.template_timestep}to{str('%02d' % (self.timestep_num))}.npz"), seg_final)
        if save_nrrd:
            nrrd.write(os.path.join(self.warped_seg_maps_dir, f"seg_{self.valid_set.patient_index}_{self.valid_set.template_timestep}to{self.timestep_num}.nrrd"), seg_final)

    def extract_timestep_num_from_ckpts_name(self, ckpts_path):
        # match = re.search('to(\d+)', ckpts_path)

        ckpts_specific_dir = ckpts_path.split("/")[-2]
        match = re.search(r'\d+', ckpts_specific_dir)

        if match:
            timestep_num = int(match.group())
            # timestep_num = int(match.group(1))
            return timestep_num
        return -1
        
