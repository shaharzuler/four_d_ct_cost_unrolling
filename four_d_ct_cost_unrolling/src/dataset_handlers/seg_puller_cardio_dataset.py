from dataclasses import asdict

import numpy as np
from torch.utils.data import Dataset
import torch
from scipy import ndimage 
from flow_n_corr_utils import xyz3_to_3xyz

from .data_sample import SegmentationPullerSampleArgs, SegmentationPullerSample, SegmentationPullerSampleWithConstraintsArgs, SegmentationPullerSampleWithConstraints
from ..utils.flow_utils import attach_flow_between_segs
from ..utils.torch_utils import torch_to_np


class SegmentationPullerCardioDataset(Dataset):
    def __init__(self, dataset_args:SegmentationPullerSampleArgs, sample_type:SegmentationPullerSample, normalize=True, scale_down_by:int=1)-> None:
        self.sample = sample_type(**{
            'template_image' : torch.tensor(ndimage.zoom(np.load(dataset_args.template_image_path), 1/scale_down_by)),  
            'unlabeled_image' : torch.tensor(ndimage.zoom(np.load(dataset_args.unlabeled_image_path), 1/scale_down_by)),   
            'template_seg' : torch.tensor(ndimage.zoom(np.load(dataset_args.template_seg_path), 1/scale_down_by)), 
            'unlabeled_seg' : torch.tensor(ndimage.zoom(np.load(dataset_args.unlabeled_seg_path), 1/scale_down_by))
            })

        if dataset_args.flows_gt_path is not None:
            self.sample.flows_gt = torch.tensor(ndimage.zoom(xyz3_to_3xyz(np.load(dataset_args.flows_gt_path)), (1,1/scale_down_by,1/scale_down_by,1/scale_down_by))) /scale_down_by
            self.sample.flows_gt = torch.nan_to_num(self.sample.flows_gt) 
        else:
            self.sample.flows_gt = torch.tensor([])
        if normalize:
            min_ = min(self.sample.template_image.min(), self.sample.unlabeled_image.min())
            max_ = max(self.sample.template_image.max(), self.sample.unlabeled_image.max())
            self.sample.template_image  = self.min_max_norm(self.sample.template_image,  min_, max_)
            self.sample.unlabeled_image = self.min_max_norm(self.sample.unlabeled_image, min_, max_)

        self.sample_dict = asdict(self.sample)

    def min_max_norm(self, img:torch.Tensor, min_:float, max_:float) -> torch.Tensor:
        return (img-min_)/(max_-min_)


    def __getitem__(self, i:int) -> SegmentationPullerSample:
        return self.sample_dict 

    def __len__(self) -> int:
        return 1 


class SegmentationPullerCardioDatasetWithConstraints(SegmentationPullerCardioDataset): # this lean version only supports overfit. for the full version go to https://github.com/gallif/_4DCTCostUnrolling
    def __init__(self, dataset_args:SegmentationPullerSampleWithConstraintsArgs, scale_down_by:int=1)-> None:
        super().__init__(dataset_args=dataset_args, sample_type=SegmentationPullerSampleWithConstraints, scale_down_by=scale_down_by) 
        two_d_constraints_arr = ndimage.zoom(np.load(dataset_args.two_d_constraints_path), (1/scale_down_by,1/scale_down_by,1/scale_down_by,1),order=0) / scale_down_by# a np arr shape x,y,z,3 with mostly np.Nans and some floats. 
        two_d_constraints_raw = attach_flow_between_segs(two_d_constraints_arr.copy(), torch_to_np(self.sample.template_seg).copy())
        two_d_constraints_processed = self.preprocess_2d_constraints(two_d_constraints_raw.copy()) 
        two_d_constraints_raw_with_nans_transposed = xyz3_to_3xyz(two_d_constraints_raw.copy()) 
        self.sample.two_d_constraints_with_nans = xyz3_to_3xyz(two_d_constraints_processed.copy()) 
        self.sample.two_d_constraints = np.nan_to_num(self.sample.two_d_constraints_with_nans.copy(), copy=True)
        self.sample.two_d_constraints_mask = np.sum(~np.isnan(two_d_constraints_raw_with_nans_transposed), axis=0).astype(bool) #the mask is usef to calculate error over the surface and shpuld be of the raw surface component rather than the processed one
        self.sample_dict = asdict(self.sample)

    def preprocess_2d_constraints(self, two_d_constraints:np.ndarray, preprocess_args:dict=None) -> np.ndarray:
        """ Here we can add more preprocessing such as blurring, thickening etc """
        k_interpolate_sparse_constraints_nn = 26
        if k_interpolate_sparse_constraints_nn > 1:
            for axis in range(two_d_constraints.shape[-1]):
                two_d_constraints = self._interpolate_knn_axis(k_interpolate_sparse_constraints_nn, two_d_constraints.copy(), axis)
        
        return two_d_constraints

    def _interpolate_knn_axis(self, k_interpolate_sparse_constraints_nn:int, voxelized_flow:np.array, axis:int) -> np.array: #TODO use the one that is in flow_n_corr
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree
        data_mask = np.isfinite(voxelized_flow[:,:,:,axis] )
        data_coords = np.array(np.where(data_mask)).T
        data_values = voxelized_flow[:,:,:,axis][data_mask]

        nan_mask = np.isnan(voxelized_flow[:,:,:,axis] )
        nan_coords = np.array(np.where(nan_mask)).T

        kdtree = cKDTree(nan_coords)
        distances, nn_indices = kdtree.query(data_coords, k=k_interpolate_sparse_constraints_nn)
        nan_coords_for_interp = nan_coords[nn_indices].reshape(-1,3)

        interpolated_values = griddata(data_coords, data_values, nan_coords_for_interp , method='linear')
        voxelized_flow[nan_coords_for_interp[:,0], nan_coords_for_interp[:,1], nan_coords_for_interp[:,2], axis ] = interpolated_values
        return voxelized_flow



   