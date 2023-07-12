from dataclasses import asdict

import numpy as np
from torch.utils.data import Dataset
import torch

from .data_sample import SegmentationPullerSampleArgs, SegmentationPullerSample, SegmentationPullerSampleWithConstraintsArgs, SegmentationPullerSampleWithConstraints
from ..utils.flow_utils import attach_flow_between_segs, xyz3_to_3xyz
from ..utils.torch_utils import torch_to_np


class SegmentationPullerCardioDataset(Dataset):
    def __init__(self, dataset_args:SegmentationPullerSampleArgs, sample_type:SegmentationPullerSample, normalize=True)-> None:
        self.sample = sample_type(**{
            'template_image' : torch.tensor(np.load(dataset_args.template_image_path)),  
            'unlabeled_image' : torch.tensor(np.load(dataset_args.unlabeled_image_path)),  
            'template_seg' : torch.tensor(np.load(dataset_args.template_seg_path)),
            'unlabeled_seg' : torch.tensor(np.load(dataset_args.unlabeled_seg_path))
            })

        if dataset_args.flows_gt_path is not None:
            self.sample.flows_gt = -torch.tensor(xyz3_to_3xyz(np.load(dataset_args.flows_gt_path)))
        else:
            self.sample.flows_gt = torch.tensor([])
        if normalize:
            min_ = min(self.sample.template_image.min(), self.sample.unlabeled_image.min())
            max_ = max(self.sample.template_image.max(), self.sample.unlabeled_image.max())
            self.sample.template_image  = self.min_max_norm(self.sample.template_image,  min_, max_)
            self.sample.unlabeled_image = self.min_max_norm(self.sample.unlabeled_image, min_, max_)

        self.sample_dict = asdict(self.sample)

    def min_max_norm(self, img:torch.tensor, min_:float, max_:float) -> torch.tensor:
        return (img-min_)/(max_-min_)


    def __getitem__(self, i:int) -> SegmentationPullerSample:
        return self.sample_dict 

    def __len__(self) -> int:
        return 1 


class SegmentationPullerCardioDatasetWithConstraints(SegmentationPullerCardioDataset): # this lean version only supports overfit. for the full version go to https://github.com/gallif/_4DCTCostUnrolling
    def __init__(self, dataset_args:SegmentationPullerSampleWithConstraintsArgs)-> None:
        super().__init__(dataset_args=dataset_args, sample_type=SegmentationPullerSampleWithConstraints) 
        two_d_constraints_arr = np.load(dataset_args.two_d_constraints_path) # a np arr shape x,y,z,3 with mostly np.Nans and some floats. 
        two_d_constraints = attach_flow_between_segs(two_d_constraints_arr, torch_to_np(self.sample.template_seg))
        two_d_constraints = self.preprocess_2d_constraints(two_d_constraints) 
        self.sample.two_d_constraints_with_nans = xyz3_to_3xyz(two_d_constraints) 
        self.sample.two_d_constraints = np.nan_to_num(self.sample.two_d_constraints_with_nans)
        self.sample.two_d_constraints_mask = ~np.isnan(self.sample.two_d_constraints_with_nans)[0] 
        self.sample_dict = asdict(self.sample)

    def preprocess_2d_constraints(self, two_d_constraints:np.array) -> np.array:
        """ Here we can add more preprocessing such as blurring, thickening etc """
        return two_d_constraints
   