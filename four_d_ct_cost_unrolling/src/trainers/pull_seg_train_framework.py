from .train_framework import TrainFramework
from ..utils.visualization_utils import disp_warped_img, disp_training_fig, add_mask 
from ..utils.flow_utils import flow_warp
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
import torch
import time
import scipy
import re
import nrrd
import os
from flow_vis import flow_to_color
import cv2



class PullSegmentationMapTrainFramework(TrainFramework):
    def __init__(self, train_loader, valid_loader, model, loss_func, args) -> None:
        super().__init__(train_loader, valid_loader, model, loss_func, args)
        self.reduce_loss_delay = 0
        self.old_avg_loss = 1E10
        self.max_reduce_loss_delay = args.max_reduce_loss_delay

    def _run_one_epoch(self) -> bool:
        am_batch_time, am_data_time, key_meter_names, key_meters, end = self._init_epoch()

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.args.epoch_size:
                break

            for n,d in enumerate(data):
                prepared_data = self._prepare_data(d)
                am_data_time.update(time.time() - end)
                res_dict = self.model(prepared_data) 
                flows, aux = self._post_process_model_output(res_dict)
                loss, l_ph, l_sm, flow_mean, l_constraints = self._compute_loss_terms(d, prepared_data["img1"], prepared_data["img2"], prepared_data["vox_dim"], flows, aux, res_dict)
                meters = [loss, l_ph, l_sm, flow_mean, l_constraints]
                vals = [m.item() if torch.is_tensor(m) else m for m in meters]
                key_meters.update(vals, prepared_data["img1"].size(0))
                self.optimize(loss)
                am_batch_time.update(time.time() - end)
                end = time.time()

                self.update_logs(am_batch_time, am_data_time, key_meter_names, key_meters, i_step)
                if i_step % self.args.plot_freq == 0:
                    self._visualize(prepared_data, flows, n, i_step, d)
                
                self.i_iter += 1
        avg_loss=key_meters.get_avg_meter_name("Loss")
        validation_data = {
            "validate_self":{"avg_loss": avg_loss}, 
            "synt_validate":{
                "flows_pred": flows,
                "flows_gt": d.flows_gt
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




    def _visualize(self, prepared_data, preds, n, i_step, d):   # TODO REFACTOR THIS MESS
        img1_recons = flow_warp(prepared_data["img2"][0].unsqueeze(0), pred_flows.unsqueeze(0))

        p_warped_4    = disp_warped_img(prepared_data["img1"][0].detach().cpu(), img1_recons[0].detach().cpu(), prepared_data["img2"].cpu(), split_index=4   )
        p_warped_2    = disp_warped_img(prepared_data["img1"][0].detach().cpu(), img1_recons[0].detach().cpu(), prepared_data["img2"].cpu(), split_index=2   )
        p_warped_1_33 = disp_warped_img(prepared_data["img1"][0].detach().cpu(), img1_recons[0].detach().cpu(), prepared_data["img2"].cpu(), split_index=1.33)

        self.summary_writer.add_images(f'Warped_{n}_{i_step}_{"_4"}',    p_warped_4,    self.i_epoch, dataformats='NHWC')
        self.summary_writer.add_images(f'Warped_{n}_{i_step}_{"_2"}',    p_warped_2,    self.i_epoch, dataformats='NHWC')
        self.summary_writer.add_images(f'Warped_{n}_{i_step}_{"_1.33"}', p_warped_1_33, self.i_epoch, dataformats='NHWC')
    
        p_valid = disp_training_fig(prepared_data["img1"][0].detach().cpu(), prepared_data["img2"][0].detach().cpu(), pred_flows.cpu())
        flows_vis = [p_valid]

        if "unconstrained_flows_fw" in preds.keys():
            unconstrained_flow = preds['unconstrained_flows_fw'][0][0].detach().squeeze(0).cpu()
            while len(unconstrained_flow.shape) > 4:
                unconstrained_flow = unconstrained_flow.squeeze(0)
            indices = np.array(unconstrained_flow.shape[1:]) // 2
            i, j, k = indices
            slice_x_flow_unconstrained = unconstrained_flow[1:3, i, :, :]
            slice_x_flow_col = flow_to_color(slice_x_flow_unconstrained.permute([1, 2, 0]).numpy(), convert_to_bgr=False)
            slice_y_flow_unconstrained = torch.stack((unconstrained_flow[0, :, j, :], unconstrained_flow[2, :, j, :]))
            slice_y_flow_col = flow_to_color(slice_y_flow_unconstrained.permute([1, 2, 0]).numpy(), convert_to_bgr=False)
            slice_z_flow_unconstrained = unconstrained_flow[0:2, :, :, k]
            slice_z_flow_col = flow_to_color(slice_z_flow_unconstrained.permute([1, 2, 0]).numpy(), convert_to_bgr=False)
            flows12 = [np.transpose(fl_,(2,0,1)) for fl_ in [slice_x_flow_col, slice_y_flow_col, slice_z_flow_col]]
            flows12_img = np.concatenate(flows12,axis=2)
            flows_vis.append(np.expand_dims(flows12_img,axis=0))

        if "2d_constraints" in preds.keys():
            two_d_constraints = preds['2d_constraints']
            while len(two_d_constraints.shape) > 4:
                two_d_constraints = two_d_constraints.squeeze(0)
            indices = np.array(two_d_constraints.shape[1:]) // 2
            i, j, k = indices
            slice_x_flow_constraints = two_d_constraints[1:3, i, :, :]
            slice_x_flow_col = flow_to_color(slice_x_flow_constraints.permute([1, 2, 0]).numpy(), convert_to_bgr=False)
            slice_y_flow_constraints = torch.stack((two_d_constraints[0, :, j, :], two_d_constraints[2, :, j, :]))
            slice_y_flow_col = flow_to_color(slice_y_flow_constraints.permute([1, 2, 0]).numpy(), convert_to_bgr=False)
            slice_z_flow_constraints = two_d_constraints[0:2, :, :, k]
            slice_z_flow_col = flow_to_color(slice_z_flow_constraints.permute([1, 2, 0]).numpy(), convert_to_bgr=False)
            two_d_constraints_flow = [np.transpose(fl_,(2,0,1)) for fl_ in [slice_x_flow_col, slice_y_flow_col, slice_z_flow_col]]
            two_d_constraints_flow_img = np.concatenate(two_d_constraints_flow,axis=2)
            flows_vis.append(np.expand_dims(two_d_constraints_flow_img,axis=0))
        
        flows_vis_img = np.concatenate(flows_vis, axis=2)
        
        self.summary_writer.add_images(f'Sample_{i_step}', flows_vis_img, self.i_epoch, dataformats='NCHW')

        template_seg_map = d["template_seg_map"]
        seg_reconst = flow_warp(template_seg_map.unsqueeze(0).float(), pred_flows.cpu().unsqueeze(0))
        p_warped_4_seg = add_mask(p_warped_4, template_seg_map, seg_reconst, split_index=4)
        p_warped_2_seg =  add_mask(p_warped_2, template_seg_map, seg_reconst, split_index=2)
        p_warped_1_33_seg = add_mask(p_warped_1_33, template_seg_map, seg_reconst, split_index=1.33)

        self.summary_writer.add_images(f'Warped_seg_{n}_{i_step}_{"_4"}'   , p_warped_4_seg,    self.i_epoch, dataformats='NHWC')
        self.summary_writer.add_images(f'Warped_seg_{n}_{i_step}_{"_2"}'   , p_warped_2_seg,    self.i_epoch, dataformats='NHWC')
        self.summary_writer.add_images(f'Warped_seg_{n}_{i_step}_{"_1.33"}', p_warped_1_33_seg, self.i_epoch, dataformats='NHWC')

        img1=prepared_data["img1"][0].detach().cpu()
        seg = d["template_seg_map"][0,:,:,:]
        seg_target = d["target_seg_map"][0,:,:,:] #TODO TEMP

        while len(img1.shape) > 3:
            img1 = img1.squeeze(0)
        while len(pred_flows.shape) > 4:
            pred_flows = pred_flows.squeeze(0)
        indices = np.array(img1.shape) // 2
        i, j, k = indices

        slice_x_1 = cv2.cvtColor(img1[i, :, :],cv2.COLOR_GRAY2RGB)
        slice_y_1 = cv2.cvtColor(img1[:, j, :],cv2.COLOR_GRAY2RGB)
        slice_z_1 = cv2.cvtColor(img1[:, :, k],cv2.COLOR_GRAY2RGB)
        mask_x_1 = seg[i, :, :]
        mask_y_1 = seg[:, j, :]
        mask_z_1 = seg[:, :, k]

        mask_x_1_target = seg_target[i, :, :] #TODO TEMP
        mask_y_1_target = seg_target[:, j, :] #TODO TEMP
        mask_z_1_target = seg_target[:, :, k] #TODO TEMP
        
        slice_x_flow_constrained = (pred_flows.cpu()[1:3, i, :, :]).numpy()
        slice_y_flow_constrained = (torch.stack((pred_flows.cpu()[0, :, j, :], pred_flows.cpu()[2, :, j, :]))).numpy()
        slice_z_flow_constrained = (pred_flows.cpu()[0:2, :, :, k]).numpy()
        if "2d_constraints" in preds.keys():
            slice_x_flow_unconstrained = (unconstrained_flow.cpu()[1:3, i, :, :]).numpy()
            slice_y_flow_unconstrained = (torch.stack((unconstrained_flow.cpu()[0, :, j, :], unconstrained_flow.cpu()[2, :, j, :]))).numpy()
            slice_z_flow_unconstrained = (unconstrained_flow.cpu()[0:2, :, :, k]).numpy()
        contours_x, hierarchy_x = cv2.findContours(image=mask_x_1.astype(np.uint8) , mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contours_y, hierarchy_y = cv2.findContours(image=mask_y_1.astype(np.uint8) , mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contours_z, hierarchy_z = cv2.findContours(image=mask_z_1.astype(np.uint8) , mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        contours_x_target, hierarchy_x = cv2.findContours(image=mask_x_1_target.astype(np.uint8) , mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) #TODO TEMP
        contours_y_target, hierarchy_y = cv2.findContours(image=mask_y_1_target.astype(np.uint8) , mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) #TODO TEMP
        contours_z_target, hierarchy_z = cv2.findContours(image=mask_z_1_target.astype(np.uint8) , mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) #TODO TEMP

        c_x = np.zeros([0])
        for c in contours_x:
            if c.shape[0]>c_x.shape[0]:
                c_x = c
        c_y = np.zeros([0])
        for c in contours_y:
            if c.shape[0]>c_y.shape[0]:
                c_y = c
        c_z = np.zeros([0])
        for c in contours_z:
            if c.shape[0]>c_z.shape[0]:
                c_z = c

        c_x_target = np.zeros([0]) #TODO TEMP
        for c in contours_x_target: #TODO TEMP
            if c.shape[0]>c_x_target.shape[0]: #TODO TEMP
                c_x_target = c #TODO TEMP
        c_y_target = np.zeros([0]) #TODO TEMP
        for c in contours_y_target: #TODO TEMP
            if c.shape[0]>c_y_target.shape[0]: #TODO TEMP
                c_y_target = c #TODO TEMP
        c_z_target = np.zeros([0]) #TODO TEMP
        for c in contours_z_target: #TODO TEMP
            if c.shape[0]>c_z_target.shape[0]: #TODO TEMP
                c_z_target = c #TODO TEMP
        
        c_x = c_x[::2,:,:]
        c_y = c_y[::2,:,:]
        c_z = c_z[::2,:,:]

        slice_x_1_constrained = slice_x_1.copy()
        slice_x_1_unconstrained = slice_x_1.copy()
        slice_x_1_constraints = slice_x_1.copy()
        arrow_scale_factor=1
        equal_arrow_length = False
        for xx in c_x_target: #TODO TEMP
            slice_x_1_constraints[xx[0,1], xx[0,0],:] = [0.,0.,0.99] #TODO TEMP
        for xx in c_x: #TODO TEMP
            slice_x_1_constraints[xx[0,1], xx[0,0],:] = [0.99,0.,0.] #TODO TEMP
        for c in c_x:
            start = c[-1]
            delta_constrained = slice_x_flow_constrained[:,c[0,0], c[0,1]]#*5
            if equal_arrow_length:
                delta_constrained /= np.linalg.norm(delta_constrained, 2)
            end_constrained = np.round(start+delta_constrained*arrow_scale_factor).astype(start.dtype)
            if "2d_constraints" in preds.keys():
                delta_unconstrained = slice_x_flow_unconstrained[:,c[0,0], c[0,1]]#*5
                if equal_arrow_length:
                    delta_unconstrained /= np.linalg.norm(delta_unconstrained, 2)
                end_unconstrained = np.round(start+delta_unconstrained*arrow_scale_factor).astype(start.dtype)
                delta_constraints = slice_x_flow_constraints.cpu().numpy()[:,c[0,0], c[0,1]]#*5
                if equal_arrow_length:
                    if np.linalg.norm(delta_constraints, 2)>0:
                        delta_constraints /= np.linalg.norm(delta_constraints, 2)
                end_constraints = np.round(start+delta_constraints*arrow_scale_factor).astype(start.dtype)

                slice_x_1_unconstrained = cv2.arrowedLine(slice_x_1_unconstrained,(start[0],start[1]),(end_unconstrained[0],end_unconstrained[1]),color=(0,0,0),thickness=1)
                slice_x_1_constraints = cv2.arrowedLine(slice_x_1_constraints,(start[0],start[1]),(end_constraints[0],end_constraints[1]),color=(0,0,0),thickness=1)                 
            slice_x_1_constrained = cv2.arrowedLine(slice_x_1_constrained,(start[0],start[1]),(end_constrained[0],end_constrained[1]),color=(0,0,0),thickness=1)



        slice_y_1_constrained = slice_y_1.copy()
        slice_y_1_unconstrained = slice_y_1.copy()
        slice_y_1_constraints = slice_y_1.copy()
        for yy in c_y_target: #TODO TEMP
            slice_y_1_constraints[yy[0,1], yy[0,0],:] = [0.,0.,0.99] #TODO TEMP
        for yy in c_y: #TODO TEMP
            slice_y_1_constraints[yy[0,1], yy[0,0],:] = [0.99,0.,0.] #TODO TEMP
        for c in c_y:
            start = c[-1]
            delta_constrained = slice_y_flow_constrained[:,c[0,0], c[0,1]]#*5
            if equal_arrow_length:
                delta_constrained /= np.linalg.norm(delta_constrained, 2)
            end_constrained = np.round(start+delta_constrained*arrow_scale_factor).astype(start.dtype)
            if "2d_constraints" in preds.keys():
                delta_unconstrained = slice_y_flow_unconstrained[:,c[0,0], c[0,1]]#*5
                if equal_arrow_length:
                    delta_unconstrained /= np.linalg.norm(delta_unconstrained, 2)
                end_unconstrained = np.round(start+delta_unconstrained*arrow_scale_factor).astype(start.dtype)
                delta_constraints = slice_y_flow_constraints.cpu().numpy()[:,c[0,0], c[0,1]]#*5
                if equal_arrow_length:
                    if np.linalg.norm(delta_constraints, 2)>0:
                        delta_constraints /= np.linalg.norm(delta_constraints, 2)
                end_constraints = np.round(start+delta_constraints*arrow_scale_factor).astype(start.dtype)
            
                slice_y_1_unconstrained = cv2.arrowedLine(slice_y_1_unconstrained,(start[0],start[1]),(end_unconstrained[0],end_unconstrained[1]),color=(0,0,0),thickness=1)
                slice_y_1_constraints = cv2.arrowedLine(slice_y_1_constraints,(start[0],start[1]),(end_constraints[0],end_constraints[1]),color=(0,0,0),thickness=1)
            slice_y_1_constrained = cv2.arrowedLine(slice_y_1_constrained,(start[0],start[1]),(end_constrained[0],end_constrained[1]),color=(0,0,0),thickness=1)
        


        slice_z_1_constrained = slice_z_1.copy()
        slice_z_1_unconstrained = slice_z_1.copy()
        slice_z_1_constraints = slice_z_1.copy()
        for zz in c_z_target: #TODO TEMP
            slice_z_1_constraints[zz[0,1], zz[0,0],:] = [0.,0.,0.99] #TODO TEMP
        for zz in c_z: #TODO TEMP
            slice_z_1_constraints[zz[0,1], zz[0,0],:] = [0.99,0.,0.] #TODO TEMP
        for c in c_z:
            start = c[-1]
            delta_constrained = slice_z_flow_constrained[:,c[0,0], c[0,1]]#*7
            if equal_arrow_length:
                delta_constrained /= np.linalg.norm(delta_constrained, 2)
            end_constrained = (np.trunc(start + delta_constrained*arrow_scale_factor) + np.sign(start+delta_constrained*arrow_scale_factor)).astype(start.dtype)
            if "2d_constraints" in preds.keys():
                delta_unconstrained = slice_z_flow_unconstrained[:,c[0,0], c[0,1]]#*7
                if equal_arrow_length:
                    delta_unconstrained /= np.linalg.norm(delta_unconstrained, 2)
                end_unconstrained = (np.trunc(start+delta_unconstrained*arrow_scale_factor) + np.sign(start+delta_constrained*arrow_scale_factor)).astype(start.dtype)
                delta_constraints = slice_z_flow_constraints.cpu().numpy()[:,c[0,0], c[0,1]]#*7
                if equal_arrow_length:
                    if np.linalg.norm(delta_constraints, 2)>0:
                        delta_constraints /= np.linalg.norm(delta_constraints, 2)
                end_constraints = (np.trunc(start+delta_constraints*arrow_scale_factor) + np.sign(start+delta_constrained*arrow_scale_factor)).astype(start.dtype)
            
                slice_z_1_unconstrained = cv2.arrowedLine(slice_z_1_unconstrained,(start[0],start[1]),(end_unconstrained[0],end_unconstrained[1]),color=(0,0,0),thickness=1)
                slice_z_1_constraints = cv2.arrowedLine(slice_z_1_constraints,(start[0],start[1]),(end_constraints[0],end_constraints[1]),color=(0,0,0),thickness=1)
            slice_z_1_constrained = cv2.arrowedLine(slice_z_1_constrained,(start[0],start[1]),(end_constrained[0],end_constrained[1]),color=(0,0,0),thickness=1)

        if "2d_constraints" in preds.keys():
            all_flow_arrowed_vis = np.concatenate(
                [
                np.concatenate([slice_x_1_constrained, slice_y_1_constrained, slice_z_1_constrained],axis=1),
                np.concatenate([slice_x_1_unconstrained, slice_y_1_unconstrained, slice_z_1_unconstrained],axis=1),
                np.concatenate([slice_x_1_constraints, slice_y_1_constraints, slice_z_1_constraints],axis=1)
                ], axis=0)
        else:
            all_flow_arrowed_vis = np.concatenate(
                [
                np.concatenate([slice_x_1_constrained, slice_y_1_constrained, slice_z_1_constrained],axis=1)
                ], axis=0)

        all_flow_arrowed_vis = np.expand_dims(np.transpose(all_flow_arrowed_vis, (2,0,1)),0)
        self.summary_writer.add_images('Sample_flows_{}'.format(i_step), all_flow_arrowed_vis, self.i_epoch, dataformats='NCHW')





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
                    pred_flows[axis,:,:,:] = torch.tensor(scipy.ndimage.median_filter(input=pred_flows[axis,:,:].cpu().numpy(), size=7))
                    
            self.save_warped_mask(d, pred_flows) 

    def save_warped_mask(self, data, pred_flows, save_nrrd=False): #TODO REFACTOR COMPLETELY
        template_seg_map = data["template_seg_map"] # template_seg_map.shape is now 1, 192, 192, 192
        seg_reconst = flow_warp(template_seg_map.unsqueeze(0).float(), pred_flows.cpu().unsqueeze(0)) # seg_reconst.shape is now 1, 1, 192, 192, 192. floats!
        # TODO convert seg_reconst to int
        seg = seg_reconst.cpu().numpy()[0,0,:,:,:]

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
        
