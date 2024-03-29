from easydict import EasyDict


from four_d_ct_cost_unrolling import overfit_backbone, infer_backbone, overfit_w_constraints, infer_w_constraints, overfit_w_seg, infer_w_seg
from four_d_ct_cost_unrolling import get_default_backbone_config, get_default_w_segmentation_config, get_default_w_constraints_config, get_checkpoints_path, get_default_checkpoints_path


args = get_default_backbone_config()
args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 10
args["valid_type"] = "basic+synthetic"
args["w_sm_scales"] = [0,0,0,0,0]
args["visualization_arrow_scale_factor"] = 1
args["cuda_device"] = 1
args["scale_down_by"] = 2
args["loss"] = "unflow+segmentation"
args["metric_for_early_stopping"] = "shell_volume_error" 

args["load"] = get_default_checkpoints_path()


backbone_model_output_path = overfit_backbone(
    template_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
    unlabeled_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    template_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    template_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    flows_gt_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/flow_for_image_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    error_radial_coordinates_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/error_radial_coordinates_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy",
    error_circumferential_coordinates_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/error_circumferential_coordinates_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy",
    error_longitudinal_coordinates_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/error_longitudinal_coordinates_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy",
    voxelized_normals_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/outputs_20240112_020626/normals.npy",
    args=EasyDict(args)
    )

args["load"] = get_checkpoints_path(backbone_model_output_path)

backbone_inference_output_path = infer_backbone(
    template_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
    unlabeled_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    template_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    template_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    flows_gt_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/flow_for_image_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    args=EasyDict(args)
    )
args = get_default_w_segmentation_config()

args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 10
args["valid_type"] = "basic+synthetic"
args["w_sm_scales"] = [0,0,0,0,0]
args["visualization_arrow_scale_factor"] = 1
args["cuda_device"] = 1
args["scale_down_by"] = 2

args["w_seg_scales"] =  [0.1, 0.1, 0.1, 0.1, 0.1 ] #[0.4, 0.4, 0.4, 0.4, 0.4 ] #

args["metric_for_early_stopping"] = "shell_volume_error" 


seg_model_output_path = overfit_w_seg(
    template_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
    unlabeled_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    template_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    template_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    flows_gt_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/flow_for_image_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    error_radial_coordinates_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/error_radial_coordinates_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy",
    error_circumferential_coordinates_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/error_circumferential_coordinates_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy",
    error_longitudinal_coordinates_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/error_longitudinal_coordinates_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy",
    voxelized_normals_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/outputs_20240112_020626/normals.npy",
    args=EasyDict(args)
    )

args["load"] = get_checkpoints_path(seg_model_output_path)

seg_inference_output_path = infer_w_seg(
    template_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
    unlabeled_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    template_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    template_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    args=EasyDict(args)
    )

# backbone_model_output_path = "/home/shahar/cardio_corr/outputs/magix/outputs_20230910_223618/outputs_backbone_training_20230910_223633"

args = get_default_w_constraints_config()
args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 1000
args["valid_type"] = "basic+synthetic"
args["load"] = get_checkpoints_path(backbone_model_output_path) #"/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/4dct_costunrolling_model_best.pth.tar"#
args["scale_down_by"] = 2
args["w_sm_scales"] = [0.0, 0, 0, 0, 0]
args["cuda_device"] = 1
args["metric_for_early_stopping"] = "shell_volume_error" 


constraints_model_output_path = overfit_w_constraints(
    template_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
    unlabeled_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    template_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    template_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    flows_gt_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/flow_for_image_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    two_d_constraints_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/outputs_20240112_020626/constraints.npy",
    error_radial_coordinates_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/error_radial_coordinates_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy",
    error_circumferential_coordinates_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/error_circumferential_coordinates_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy",
    error_longitudinal_coordinates_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/error_longitudinal_coordinates_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy",
    voxelized_normals_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/outputs_20240112_020626/normals.npy",
    args=EasyDict(args)
    )

args["load"] = get_checkpoints_path(constraints_model_output_path)

constraints_inference_output_path = infer_w_constraints(
    template_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
    unlabeled_image_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    template_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_LV_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    template_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_shell_seg_path='/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/dataset_tot_torsion_100_torsion_version_4/thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/extra_mask_orig_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    two_d_constraints_path="/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_100_torsion_version_4/outputs_20240112_020626/constraints.npy",
    save_mask=False,
    args=EasyDict(args)
    )

print("completed dummy run")

# future work: 
# save 2d_constraints with model weights so that inference will not require the constraints as an input