from dataclasses import dataclass

import numpy as np

@dataclass
class SegmentationPullerSampleArgs:
    template_image_path : str
    unlabeled_image_path : str
    template_seg_path : str = None
    unlabeled_seg_path : str = None
    flows_gt_path : str = None

@dataclass
class SegmentationPullerSample:
    template_image : np.array
    unlabeled_image : np.array
    template_seg : np.array = None
    unlabeled_seg : np.array = None
    flows_gt : np.array = None

@dataclass
class SegmentationPullerSampleWithConstraintsArgs(SegmentationPullerSampleArgs):
    two_d_constraints_path : str = None
    
@dataclass
class SegmentationPullerSampleWithConstraints(SegmentationPullerSample):
    two_d_constraints : np.array = None
    two_d_constraints_with_nans : np.array = None
    two_d_constraints_mask : np.array = None

