from easydict import EasyDict


from four_d_ct_cost_unrolling import overfit_backbone, infer_backbone, overfit_w_constraints, infer_w_constraints
from four_d_ct_cost_unrolling import get_default_backbone_config, get_default_w_constraints_config, get_checkpoints_path


args = get_default_backbone_config()
args["save_iter"] = 2
args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 10
args["valid_type"] = "basic+synthetic"

args["w_sm_scales"] = [0,0,0,0,0]
args["visualization_arrow_scale_factor"] = 1
args["cuda_device"] = 1
args["scale_down_by"] = 2


backbone_model_output_path = overfit_backbone(
    template_image_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_image_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    template_seg_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_skewed_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_seg_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    flows_gt_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/flow_for_image_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    args=EasyDict(args)
    )

args["load"] = get_checkpoints_path(backbone_model_output_path)

backbone_inference_output_path = infer_backbone(
    template_image_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_image_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    template_seg_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_skewed_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_seg_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    args=EasyDict(args)
    )

# backbone_model_output_path = "/home/shahar/cardio_corr/outputs/magix/outputs_20230910_223618/outputs_backbone_training_20230910_223633"

args = get_default_w_constraints_config()
args["save_iter"] = 2
args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 50
args["valid_type"] = "basic+synthetic"
args["load"] = get_checkpoints_path(backbone_model_output_path) #"/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/4dct_costunrolling_model_best.pth.tar"#
args["scale_down_by"] = 2
args["w_sm_scales"] = [0.0, 0, 0, 0, 0]
args["cuda_device"] = 1

constraints_model_output_path = overfit_w_constraints(
    template_image_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_image_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    template_seg_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_skewed_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_seg_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',    
    flows_gt_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/flow_for_image_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    args=EasyDict(args)
    )

args["load"] = get_checkpoints_path(constraints_model_output_path)

constraints_inference_output_path = infer_w_constraints(
    template_image_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_image_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    template_seg_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_skewed_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    unlabeled_seg_path='/home/shahar/cardio_corr/outputs/magix/synthetic_dataset152/thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_49.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy', 
    two_d_constraints_path="/home/shahar/cardio_corr/outputs/magix/outputs_20230910_223618/constraints.npy",
    save_mask=False,
    args=EasyDict(args)
    )

print("completed dummy run")

# future work: 
# save 2d_constraints with model weights so that inference will not require the constraints as an input