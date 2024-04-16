 # this lean version only supports overfit. for the full version go to https://github.com/gallif/_4DCTCostUnrolling

from typing import Dict
from three_d_data_manager import write_config_file

from .losses.get_loss import get_loss
from .trainers.pull_seg_train_framework import PullSegmentationMapTrainFramework
from .trainers.pull_seg_train_framework_w_seg_training import PullSegmentationMapTrainFrameworkWithSegmentation

from .trainers.pull_seg_train_framework_w_2d_constraints import PullSegmentationMapTrainFrameworkWith2dConstraints
from .dataset_handlers.seg_puller_cardio_dataset import SegmentationPullerCardioDataset, SegmentationPullerCardioDatasetWithConstraints
from .dataset_handlers.data_sample import SegmentationPullerSampleArgs, SegmentationPullerSampleWithConstraintsArgs, SegmentationPullerSample
from .models.pwc3d import PWC3D
from .models.pwc3d_w_2d_constraints import PWC3Dw2dConstraints



def overfit_backbone(
    template_image_path:str, unlabeled_image_path:str, 
    template_LV_seg_path:str, unlabeled_LV_seg_path:str, 
    template_shell_seg_path:str, unlabeled_shell_seg_path:str, 
    flows_gt_path:str=None, 
    error_radial_coordinates_path:str=None,
    error_circumferential_coordinates_path:str=None,
    error_longitudinal_coordinates_path:str=None,
    voxelized_normals_path:str=None,
    args:Dict=None) -> str: 

    data_sample_args = SegmentationPullerSampleArgs(
        template_image_path, unlabeled_image_path, 
        template_LV_seg_path, unlabeled_LV_seg_path, 
        template_shell_seg_path, unlabeled_shell_seg_path,
        flows_gt_path, 
        error_radial_coordinates_path,
        error_circumferential_coordinates_path, 
        error_longitudinal_coordinates_path,
        voxelized_normals_path,
        args.num_pixels_validate_inside_seg, args.num_pixels_validate_outside_seg)
    train_set = SegmentationPullerCardioDataset(data_sample_args, sample_type=SegmentationPullerSample, scale_down_by=args.scale_down_by)
    model = PWC3D(args)
    loss = get_loss(args)
    trainer = PullSegmentationMapTrainFramework(train_set, model, loss, args)
    output_path = trainer.train(args.cuda_device)
    return output_path

def infer_backbone(
    template_image_path:str, unlabeled_image_path:str, 
    template_LV_seg_path:str, unlabeled_LV_seg_path:str, 
    template_shell_seg_path:str, unlabeled_shell_seg_path:str, 
    flows_gt_path:str=None, args:Dict=None):

    data_sample_args = SegmentationPullerSampleArgs(
        template_image_path, unlabeled_image_path, 
        template_LV_seg_path, unlabeled_LV_seg_path, 
        template_shell_seg_path, unlabeled_shell_seg_path,
        flows_gt_path)
    train_set = SegmentationPullerCardioDataset(data_sample_args, sample_type=SegmentationPullerSample, scale_down_by=args.scale_down_by)
    model = PWC3D(args)
    trainer = PullSegmentationMapTrainFramework(train_set, model, None, args)
    output_path = trainer.infer(args.cuda_device)
    return output_path
    

def validate_backbone(
    template_image_path:str, unlabeled_image_path:str, 
    template_LV_seg_path:str, unlabeled_LV_seg_path:str, 
    template_shell_seg_path:str, unlabeled_shell_seg_path:str, 
    flows_gt_path:str=None, 
    error_radial_coordinates_path:str=None,
    error_circumferential_coordinates_path:str=None,
    error_longitudinal_coordinates_path:str=None,
    voxelized_normals_path:str=None,
    args:Dict=None) -> str: 

    data_sample_args = SegmentationPullerSampleArgs(
        template_image_path, unlabeled_image_path, 
        template_LV_seg_path, unlabeled_LV_seg_path, 
        template_shell_seg_path, unlabeled_shell_seg_path,
        flows_gt_path, 
        error_radial_coordinates_path,
        error_circumferential_coordinates_path, 
        error_longitudinal_coordinates_path,
        voxelized_normals_path,
        args.num_pixels_validate_inside_seg, args.num_pixels_validate_outside_seg)
    train_set = SegmentationPullerCardioDataset(data_sample_args, sample_type=SegmentationPullerSample, scale_down_by=args.scale_down_by)
    model = PWC3D(args)
    loss = get_loss(args)
    trainer = PullSegmentationMapTrainFramework(train_set, model, loss, args)
    output_path = trainer.infer(args.cuda_device) #?
    current_errors = trainer.run_validation()
    return current_errors

def overfit_w_seg(
    template_image_path:str, unlabeled_image_path:str, 
    template_LV_seg_path:str, unlabeled_LV_seg_path:str, 
    template_shell_seg_path:str, unlabeled_shell_seg_path:str, 
    flows_gt_path:str=None, 
    error_radial_coordinates_path:str=None,
    error_circumferential_coordinates_path:str=None,
    error_longitudinal_coordinates_path:str=None,
    voxelized_normals_path:str=None,
    args:Dict=None) -> str: 

    data_sample_args = SegmentationPullerSampleArgs(
        template_image_path, unlabeled_image_path, 
        template_LV_seg_path, unlabeled_LV_seg_path, 
        template_shell_seg_path, unlabeled_shell_seg_path,
        flows_gt_path, 
        error_radial_coordinates_path,
        error_circumferential_coordinates_path, 
        error_longitudinal_coordinates_path,
        voxelized_normals_path,
        args.num_pixels_validate_inside_seg, args.num_pixels_validate_outside_seg)
    train_set = SegmentationPullerCardioDataset(data_sample_args, sample_type=SegmentationPullerSample, scale_down_by=args.scale_down_by)
    model = PWC3D(args)
    loss = get_loss(args)
    trainer = PullSegmentationMapTrainFrameworkWithSegmentation(train_set, model, loss, args)
    output_path = trainer.train(args.cuda_device)
    return output_path

def infer_w_seg(
    template_image_path:str, unlabeled_image_path:str, 
    template_LV_seg_path:str, unlabeled_LV_seg_path:str, 
    template_shell_seg_path:str, unlabeled_shell_seg_path:str, 
    flows_gt_path:str=None, args:Dict=None):

    data_sample_args = SegmentationPullerSampleArgs(
        template_image_path, unlabeled_image_path, 
        template_LV_seg_path, unlabeled_LV_seg_path, 
        template_shell_seg_path, unlabeled_shell_seg_path,
        flows_gt_path)
    train_set = SegmentationPullerCardioDataset(data_sample_args, sample_type=SegmentationPullerSample, scale_down_by=args.scale_down_by)
    model = PWC3D(args)
    trainer = PullSegmentationMapTrainFrameworkWithSegmentation(train_set, model, None, args)
    output_path = trainer.infer(args.cuda_device)
    return output_path


def validate_w_seg(
    template_image_path:str, unlabeled_image_path:str, 
    template_LV_seg_path:str, unlabeled_LV_seg_path:str, 
    template_shell_seg_path:str, unlabeled_shell_seg_path:str, 
    flows_gt_path:str=None, 
    error_radial_coordinates_path:str=None,
    error_circumferential_coordinates_path:str=None,
    error_longitudinal_coordinates_path:str=None,
    voxelized_normals_path:str=None,
    args:Dict=None) -> str: 

    data_sample_args = SegmentationPullerSampleArgs(
        template_image_path, unlabeled_image_path, 
        template_LV_seg_path, unlabeled_LV_seg_path, 
        template_shell_seg_path, unlabeled_shell_seg_path,
        flows_gt_path, 
        error_radial_coordinates_path,
        error_circumferential_coordinates_path, 
        error_longitudinal_coordinates_path,
        voxelized_normals_path,
        args.num_pixels_validate_inside_seg, args.num_pixels_validate_outside_seg)
    train_set = SegmentationPullerCardioDataset(data_sample_args, sample_type=SegmentationPullerSample, scale_down_by=args.scale_down_by)
    model = PWC3D(args)
    loss = get_loss(args)
    trainer = PullSegmentationMapTrainFrameworkWithSegmentation(train_set, model, loss, args)
    output_path = trainer.infer(args.cuda_device) #?
    current_errors = trainer.run_validation()
    return current_errors


def overfit_w_constraints(
    template_image_path:str, unlabeled_image_path:str, 
    template_LV_seg_path:str, unlabeled_LV_seg_path:str, 
    template_shell_seg_path:str, unlabeled_shell_seg_path:str, 
    two_d_constraints_path:str, 
    flows_gt_path:str=None, 
    error_radial_coordinates_path:str=None,
    error_circumferential_coordinates_path:str=None,
    error_longitudinal_coordinates_path:str=None,
    voxelized_normals_path:str=None,
    args:Dict=None) -> str:

    data_sample_args = SegmentationPullerSampleWithConstraintsArgs(
        template_image_path=template_image_path, 
        unlabeled_image_path=unlabeled_image_path, 
        template_LV_seg_path=template_LV_seg_path, 
        unlabeled_LV_seg_path=unlabeled_LV_seg_path, 
        template_shell_seg_path=template_shell_seg_path, 
        unlabeled_shell_seg_path=template_shell_seg_path,
        flows_gt_path=flows_gt_path,
        error_radial_coordinates_path=error_radial_coordinates_path,
        error_circumferential_coordinates_path=error_circumferential_coordinates_path, 
        error_longitudinal_coordinates_path=error_longitudinal_coordinates_path,
        voxelized_normals_path=voxelized_normals_path,
        num_pixels_validate_inside_seg=args.num_pixels_validate_inside_seg, 
        num_pixels_validate_outside_seg=args.num_pixels_validate_outside_seg,
        two_d_constraints_path=two_d_constraints_path,
        k_interpolate_sparse_constraints_nn=args.k_interpolate_sparse_constraints_nn)
    train_set = SegmentationPullerCardioDatasetWithConstraints(data_sample_args, scale_down_by=args.scale_down_by)
    model = PWC3Dw2dConstraints(args, train_set.sample.two_d_constraints)
    loss = get_loss(args)
    trainer = PullSegmentationMapTrainFrameworkWith2dConstraints(train_set, model, loss, args)
    output_path = trainer.train(args.cuda_device)
    return output_path

def infer_w_constraints(
    template_image_path:str, unlabeled_image_path:str, 
    template_LV_seg_path:str, unlabeled_LV_seg_path:str, 
    template_shell_seg_path:str, unlabeled_shell_seg_path:str, 
    save_mask:bool, two_d_constraints_path:str, 
    flows_gt_path:str=None, args:Dict=None) -> None:

    data_sample_args = SegmentationPullerSampleWithConstraintsArgs(template_image_path=template_image_path, 
        unlabeled_image_path=unlabeled_image_path, 
        template_LV_seg_path=template_LV_seg_path, 
        unlabeled_LV_seg_path=unlabeled_LV_seg_path, 
        template_shell_seg_path=template_shell_seg_path, 
        unlabeled_shell_seg_path=template_shell_seg_path,
        flows_gt_path=flows_gt_path,
        two_d_constraints_path=two_d_constraints_path,
         k_interpolate_sparse_constraints_nn=args.k_interpolate_sparse_constraints_nn)
    train_set = SegmentationPullerCardioDatasetWithConstraints(data_sample_args, scale_down_by=args.scale_down_by)
    model = PWC3Dw2dConstraints(args, train_set.sample.two_d_constraints)
    trainer = PullSegmentationMapTrainFrameworkWith2dConstraints(train_set, model, None, args)
    output_path = trainer.infer(args.cuda_device, save_mask=save_mask)
    return output_path

def validate_w_constraints(
    template_image_path:str, unlabeled_image_path:str, 
    template_LV_seg_path:str, unlabeled_LV_seg_path:str, 
    template_shell_seg_path:str, unlabeled_shell_seg_path:str, 
    two_d_constraints_path:str, 
    flows_gt_path:str=None, 
    error_radial_coordinates_path:str=None,
    error_circumferential_coordinates_path:str=None,
    error_longitudinal_coordinates_path:str=None,
    voxelized_normals_path:str=None,
    args:Dict=None) -> str:

    data_sample_args = SegmentationPullerSampleWithConstraintsArgs(
        template_image_path=template_image_path, 
        unlabeled_image_path=unlabeled_image_path, 
        template_LV_seg_path=template_LV_seg_path, 
        unlabeled_LV_seg_path=unlabeled_LV_seg_path, 
        template_shell_seg_path=template_shell_seg_path, 
        unlabeled_shell_seg_path=template_shell_seg_path,
        flows_gt_path=flows_gt_path,
        error_radial_coordinates_path=error_radial_coordinates_path,
        error_circumferential_coordinates_path=error_circumferential_coordinates_path, 
        error_longitudinal_coordinates_path=error_longitudinal_coordinates_path,
        voxelized_normals_path=voxelized_normals_path,
        num_pixels_validate_inside_seg=args.num_pixels_validate_inside_seg, 
        num_pixels_validate_outside_seg=args.num_pixels_validate_outside_seg,
        two_d_constraints_path=two_d_constraints_path,
        k_interpolate_sparse_constraints_nn=args.k_interpolate_sparse_constraints_nn
    )
    train_set = SegmentationPullerCardioDatasetWithConstraints(data_sample_args, scale_down_by=args.scale_down_by)
    model = PWC3Dw2dConstraints(args, train_set.sample.two_d_constraints)
    loss = get_loss(args)
    trainer = PullSegmentationMapTrainFrameworkWith2dConstraints(train_set, model, loss, args)
    output_path = trainer.infer(args.cuda_device) #?
    current_errors = trainer.run_validation()
    return current_errors
