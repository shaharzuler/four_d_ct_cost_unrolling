from dataclasses import dataclass

import numpy as np

@dataclass
class SegmentationPullerSampleArgs:
    template_image_path : str
    unlabeled_image_path : str
    template_LV_seg_path : str = None
    unlabeled_LV_seg_path : str = None
    template_shell_seg_path : str = None
    unlabeled_shell_seg_path : str = None
    flows_gt_path : str = None
    error_radial_coordinates_path : str = None
    error_circumferential_coordinates_path : str = None
    error_longitudinal_coordinates_path : str = None
    voxelized_normals_path : str = None
    num_pixels_validate_outside_seg : int = 0
    num_pixels_validate_inside_seg : int = 0

@dataclass
class SegmentationPullerSample:
    template_image : np.array
    unlabeled_image : np.array
    template_LV_seg : np.array = None
    unlabeled_LV_seg : np.array = None
    template_shell_seg : np.array = None
    unlabeled_shell_seg : np.array = None
    flows_gt : np.array = None
    error_radial_coordinates : np.array = None
    error_circumferential_coordinates : np.array = None
    error_longitudinal_coordinates : np.array = None
    voxelized_normals : np.array = None

@dataclass
class SegmentationPullerSampleWithConstraintsArgs(SegmentationPullerSampleArgs):
    two_d_constraints_path : str = None
    k_interpolate_sparse_constraints_nn : int = None
    
@dataclass
class SegmentationPullerSampleWithConstraints(SegmentationPullerSample):
    two_d_constraints : np.array = None
    two_d_constraints_with_nans : np.array = None
    two_d_constraints_mask : np.array = None

