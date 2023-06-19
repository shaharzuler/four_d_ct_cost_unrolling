from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset
import torch

@dataclass
class SegmentationPullerCardioSampleArgs:
    template_image_path : str
    unlabeled_image_path : str
    template_seg_path : str = None
    unlabeled_seg_path : str = None
    flows_gt_path : str = None

@dataclass
class SegmentationPullerCardiosampleWithConstraintsArgs(SegmentationPullerCardioSampleArgs):
    two_d_constraints_path : str=None
    
@dataclass
class SegmentationPullerSample:
    template_image : np.array
    unlabeled_image : np.array
    template_seg : np.array = None
    unlabeled_seg : np.array = None
    flows_gt : np.array = None

@dataclass
class SegmentationPullerSampleWithConstraints:
    two_d_constraints : np.array = None


class SegmentationPullerCardioDataset(Dataset): # this lean version only supports overfit. for the full version go to https://github.com/gallif/_4DCTCostUnrolling
    def __init__(self, dataset_args:SegmentationPullerCardioSampleArgs)-> None:
        self.sample = SegmentationPullerSample({
            'template_image' : torch.tensor(np.load(self.dataset_args.template_image_path)),  
            'unlabeled_image' : torch.tensor(np.load(self.dataset_args.unlabeled_image_path)),  
            'template_seg' : torch.tensor(np.load(self.dataset_args.template_seg_path)),
            'unlabeled_seg' : torch.tensor(np.load(self.dataset_args.unlabeled_seg_path))
            })
        if self.dataset_args.flows_gt_path is not None:
            self.sample.flows_gt = torch.tensor(np.load(self.dataset_args.flows_gt_path))

    def __getitem__(self, i) -> SegmentationPullerSample:
        return self.sample 

    def __len__(self):
        return 1 


class SegmentationPullerCardioDatasetWithConstraints(SegmentationPullerCardioDataset): # this lean version only supports overfit. for the full version go to https://github.com/gallif/_4DCTCostUnrolling
    def __init__(self, dataset_args:SegmentationPullerCardiosampleWithConstraintsArgs)-> None:
        super().__init__(dataset_args=dataset_args)
        self.sample.two_d_constraints = torch.tensor(np.load(self.dataset_args.two_d_constraints_path)) #TODO - orig SegmentationPullerCardioDataset has the code at the end of getitem to convert flow to constraints (enveloping, blur, extrapolate to nearest point on envelope)

  