import torch
import numpy as np
from flow_vis import flow_to_color
import cv2

from ..utils.metrics_utils import AverageMeter
from .pull_seg_train_framework import PullSegmentationMapTrainFramework



class PullSegmentationMapTrainFrameworkWith2dConstraints(PullSegmentationMapTrainFramework):
    def __init__(self, train_loader, valid_loader, model, loss_func, args) -> None:
        super().__init__(train_loader, valid_loader, model, loss_func, args)
    
    def _prepare_data(self, d):
        data = super()._prepare_data(d)
        data["2d_constraints"] = d["2d_constraints"]
        return data

    def _init_key_meters(self):
        key_meter_names = ['Loss', 'l_ph', 'l_sm', "flow_mean", "l_constraints"]
        key_meters = AverageMeter(i=len(key_meter_names), print_precision=4, names=key_meter_names)
        return key_meter_names, key_meters

    def _compute_loss_terms(self, data, img1, img2, vox_dim, flows, aux, res_dict): #TODO only pass constraints and not entire resdict or aux
        loss, l_ph, l_sm, flow_mean = super()._compute_loss_terms(img1, img2, vox_dim, flows, aux)
        l_constraints = self._get_constraints_loss(flows, res_dict, data, aux) #TODO only pass constraints and not entire resdict or aux
        loss += l_constraints

        return loss, (l_ph, l_sm, flow_mean, l_constraints)

    def _get_constraints_loss(self, flows, res_dict, data, aux):
        if "2d_constraints" in data.keys():
            aux[0]["bin_seg_mask"] = torch.unsqueeze(torch.max(torch.where(data["2d_constraints"]!=0,1,0),axis=1).values, 0) # does accidently include also where onstraints are 0. FIXME TOOD

        for loss_, module_ in self.loss_modules.items():
            if "constraints" in loss_:
                l_constraints = module_(flows, res_dict["2d_constraints"])
            else:
                l_constraints = 0.0
        return l_constraints


    def _visualize(self, data, pred_flow, i_step):   # TODO REFACTOR THIS MESS
        arrow_scale_factor=1
        equal_arrow_length = False
        if "unconstrained_flows_fw" in pred_flow.keys():
            unconstrained_flow = pred_flow['unconstrained_flows_fw'][0][0].detach().squeeze(0).cpu()
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
            flows_visualization.append(np.expand_dims(flows12_img,axis=0))

        if "2d_constraints" in pred_flow.keys():
            two_d_constraints = pred_flow['2d_constraints']
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
            flows_visualization.append(np.expand_dims(two_d_constraints_flow_img,axis=0))

        if "2d_constraints" in preds.keys():
            slice_x_flow_unconstrained = (unconstrained_flow.cpu()[1:3, i, :, :]).numpy()
            slice_y_flow_unconstrained = (torch.stack((unconstrained_flow.cpu()[0, :, j, :], unconstrained_flow.cpu()[2, :, j, :]))).numpy()
            slice_z_flow_unconstrained = (unconstrained_flow.cpu()[0:2, :, :, k]).numpy()

        slice_x_flow_constrained = (pred_flows.cpu()[1:3, i, :, :]).numpy()
        slice_y_flow_constrained = (torch.stack((pred_flows.cpu()[0, :, j, :], pred_flows.cpu()[2, :, j, :]))).numpy()
        slice_z_flow_constrained = (pred_flows.cpu()[0:2, :, :, k]).numpy()

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

class PullSegmentationMapTrainFrameworkWith2dConstraintsInference(PullSegmentationMapTrainFramework): # TODO
    pass
