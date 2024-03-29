from dataclasses import asdict
from typing import Dict

import numpy as np
from torch.utils.data import Dataset
import torch
from scipy import ndimage 
from scipy.ndimage import binary_dilation, binary_erosion
from flow_n_corr_utils import xyz3_to_3xyz, interpolate_from_flow_in_axis, min_max_norm_det_vals

from .data_sample import SegmentationPullerSampleArgs, SegmentationPullerSample, SegmentationPullerSampleWithConstraintsArgs, SegmentationPullerSampleWithConstraints
from ..utils.flow_utils import attach_flow_between_segs
from ..utils.torch_utils import torch_to_np


class SegmentationPullerCardioDataset(Dataset):
    def __init__(self, dataset_args:SegmentationPullerSampleArgs, sample_type:SegmentationPullerSample, normalize=True, scale_down_by:int=1)-> None:
        self.dataset_args = dataset_args
        self.sample = sample_type(**{
            'template_image' : torch.tensor(ndimage.zoom(np.load(dataset_args.template_image_path), 1/scale_down_by)),  
            'unlabeled_image' : torch.tensor(ndimage.zoom(np.load(dataset_args.unlabeled_image_path), 1/scale_down_by)),   
            'template_LV_seg' : torch.tensor(ndimage.zoom(np.load(dataset_args.template_LV_seg_path), 1/scale_down_by, order=0)), 
            'unlabeled_LV_seg' : torch.tensor(ndimage.zoom(np.load(dataset_args.unlabeled_LV_seg_path), 1/scale_down_by, order=0)),
            'template_shell_seg' : torch.tensor(ndimage.zoom(np.load(dataset_args.template_shell_seg_path), 1/scale_down_by, order=0)), 
            'unlabeled_shell_seg' : torch.tensor(ndimage.zoom(np.load(dataset_args.unlabeled_shell_seg_path), 1/scale_down_by, order=0))
            })
        req_shape = self.sample.template_image.shape


        if dataset_args.flows_gt_path is not None:
            arr = np.nan_to_num(xyz3_to_3xyz(np.load(dataset_args.flows_gt_path)))
            self.sample.flows_gt = torch.tensor(ndimage.zoom(arr, (1,1/scale_down_by,1/scale_down_by,1/scale_down_by))) /scale_down_by
        else:
            self.sample.flows_gt = torch.tensor([])

        if dataset_args.error_radial_coordinates_path is not None:
            arr = np.nan_to_num(xyz3_to_3xyz(np.load(dataset_args.error_radial_coordinates_path)))
            self.sample.error_radial_coordinates = torch.tensor(ndimage.zoom(arr, (1,1/scale_down_by,1/scale_down_by,1/scale_down_by))) 
        else:
            self.sample.error_radial_coordinates = torch.tensor([])
        
        if dataset_args.error_circumferential_coordinates_path is not None:
            arr = np.nan_to_num(xyz3_to_3xyz(np.load(dataset_args.error_circumferential_coordinates_path)))
            self.sample.error_circumferential_coordinates = torch.tensor(ndimage.zoom(arr, (1,1/scale_down_by,1/scale_down_by,1/scale_down_by))) 
        else:
            self.sample.error_circumferential_coordinates = torch.tensor([])
        
        if dataset_args.error_longitudinal_coordinates_path is not None:
            arr = np.nan_to_num(xyz3_to_3xyz(np.load(dataset_args.error_longitudinal_coordinates_path)))
            self.sample.error_longitudinal_coordinates = torch.tensor(ndimage.zoom(arr, (1,1/scale_down_by,1/scale_down_by,1/scale_down_by)))
            self.sample.error_longitudinal_coordinates = torch.nan_to_num(self.sample.error_longitudinal_coordinates) 
        else:
            self.sample.error_longitudinal_coordinates = torch.tensor([])

        if dataset_args.voxelized_normals_path is not None:
            arr = np.nan_to_num(xyz3_to_3xyz(np.load(dataset_args.voxelized_normals_path)))
            self.sample.voxelized_normals = torch.tensor(arr[:,::scale_down_by, ::scale_down_by, ::scale_down_by])
            if self.sample.voxelized_normals.shape[1:] != req_shape:
                self.sample.voxelized_normals = self.sample.voxelized_normals[:,:req_shape[0],:req_shape[1],:req_shape[2]]
        else:
            self.sample.voxelized_normals = torch.tensor([])

        if normalize:
            min_ = min(self.sample.template_image.min(), self.sample.unlabeled_image.min())
            max_ = max(self.sample.template_image.max(), self.sample.unlabeled_image.max())
            self.sample.template_image  = min_max_norm_det_vals(self.sample.template_image,  min_, max_)
            self.sample.unlabeled_image = min_max_norm_det_vals(self.sample.unlabeled_image, min_, max_)

        self.sample_dict = asdict(self.sample)
        
        if max(dataset_args.num_pixels_validate_inside_seg, dataset_args.num_pixels_validate_outside_seg) > 0:
            self.sample_dict["distance_validation_masks"] = self._create_distance_validation_masks(dataset_args.num_pixels_validate_inside_seg, dataset_args.num_pixels_validate_outside_seg)

    def _create_distance_validation_masks(self, num_pixels_validate_inside_seg:int, num_pixels_validate_outside_seg:int):
        validation_masks = {"in": {}, "out": {}} 
        validation_masks = self._create_distance_validation_masks_for_single_direction(num_pixels_validate_inside_seg,  validation_masks, "in")
        validation_masks = self._create_distance_validation_masks_for_single_direction(num_pixels_validate_outside_seg, validation_masks, "out")

        return validation_masks
    
    def _create_distance_validation_masks_for_single_direction(self, num_pixels_validate:int, validation_masks:Dict, in_or_out:str):
        last = (self.sample.template_LV_seg).cpu().numpy()
        morph_func = binary_dilation if in_or_out=="out" else binary_erosion
        for i in range(1, num_pixels_validate+1): 
            morphed = morph_func(last)
            mask = morphed ^ last
            validation_masks[in_or_out][i] = mask
            last = morphed
        return validation_masks

    def __getitem__(self, i:int) -> SegmentationPullerSample:
        return self.sample_dict 

    def __len__(self) -> int:
        return 1 


class SegmentationPullerCardioDatasetWithConstraints(SegmentationPullerCardioDataset): # this lean version only supports overfit. for the full version go to https://github.com/gallif/_4DCTCostUnrolling
    def __init__(self, dataset_args:SegmentationPullerSampleWithConstraintsArgs, scale_down_by:int=1)-> None:
        super().__init__(dataset_args=dataset_args, sample_type=SegmentationPullerSampleWithConstraints, scale_down_by=scale_down_by) 
        two_d_constraints_arr = np.load(dataset_args.two_d_constraints_path)[::scale_down_by, ::scale_down_by, ::scale_down_by, :] / scale_down_by # a np arr shape x,y,z,3 with mostly np.Nans and some floats. 

        two_d_constraints_raw = attach_flow_between_segs(two_d_constraints_arr.copy(), torch_to_np(self.sample.template_LV_seg).copy())
        two_d_constraints_processed = self.preprocess_2d_constraints(two_d_constraints_raw.copy()) 
        two_d_constraints_raw_with_nans_transposed = xyz3_to_3xyz(two_d_constraints_raw.copy()) 
        self.sample.two_d_constraints_with_nans = xyz3_to_3xyz(two_d_constraints_processed.copy()) 
        self.sample.two_d_constraints = np.nan_to_num(self.sample.two_d_constraints_with_nans.copy(), copy=True)
        self.sample.two_d_constraints_mask = np.sum(~np.isnan(two_d_constraints_raw_with_nans_transposed), axis=0).astype(bool) # the mask is used to calculate error over the surface and shpuld be of the raw surface component rather than the processed one
        self.sample_dict.update(asdict(self.sample))

    def preprocess_2d_constraints(self, two_d_constraints:np.ndarray, preprocess_args:dict=None) -> np.ndarray:
        """ Here we can add more preprocessing such as blurring, thickening etc """
        k_interpolate_sparse_constraints_nn = self.dataset_args.k_interpolate_sparse_constraints_nn
        if k_interpolate_sparse_constraints_nn > 1:
            for axis in range(two_d_constraints.shape[-1]):
                two_d_constraints = interpolate_from_flow_in_axis(k_interpolate_sparse_constraints_nn, two_d_constraints.copy(), axis)
        
        return two_d_constraints
