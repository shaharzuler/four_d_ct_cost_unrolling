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
args["cuda_device"] = 0


backbone_model_output_path = overfit_backbone(
    template_image_path='/home/shahar/cardio_corr/outputs/synthetic_dataset33/thetas_90.0_0.0_rs_0.9_0.9_h_0.9_geometric_mask_True_blur_radious_7/image_skewed_thetas_90.0_0.0_rs_0.9_0.9_h_0.9_geometric_mask_True_blur_radious_7.npy', 
    unlabeled_image_path='/home/shahar/cardio_corr/outputs/synthetic_dataset33/thetas_90.0_0.0_rs_0.9_0.9_h_0.9_geometric_mask_True_blur_radious_7/image_orig_thetas_90.0_0.0_rs_0.9_0.9_h_0.9_geometric_mask_True_blur_radious_7.npy', 
    template_seg_path='/home/shahar/cardio_corr/outputs/synthetic_dataset33/thetas_90.0_0.0_rs_0.9_0.9_h_0.9_geometric_mask_True_blur_radious_7/mask_skewed_thetas_90.0_0.0_rs_0.9_0.9_h_0.9_geometric_mask_True_blur_radious_7.npy', 
    unlabeled_seg_path='/home/shahar/cardio_corr/outputs/synthetic_dataset33/thetas_90.0_0.0_rs_0.9_0.9_h_0.9_geometric_mask_True_blur_radious_7/mask_orig_thetas_90.0_0.0_rs_0.9_0.9_h_0.9_geometric_mask_True_blur_radious_7.npy', 
    flows_gt_path='/home/shahar/cardio_corr/outputs/synthetic_dataset33/thetas_90.0_0.0_rs_0.9_0.9_h_0.9_geometric_mask_True_blur_radious_7/flow_for_image_thetas_90.0_0.0_rs_0.9_0.9_h_0.9_geometric_mask_True_blur_radious_7.npy', 
    args=EasyDict(args)
    )

args["load"] = get_checkpoints_path(backbone_model_output_path)

backbone_inference_output_path = infer_backbone(
    template_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    unlabeled_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
    unlabeled_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
    args=EasyDict(args)
    )

backbone_model_output_path = "/home/shahar/cardio_corr/outputs/new_runs/outputs_20230724_120732/outputs_backbone_training_20230724_120850"

args = get_default_w_constraints_config()
args["save_iter"] = 2
args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 10
args["valid_type"] = "basic+synthetic"
args["load"] = get_checkpoints_path(backbone_model_output_path)


args["w_scales"] = [0.0, 0.0, 0.0, 0.0, 0.0]
args["w_admm"] = [0.0, 0.0, 0.0, 0.0, 0.0]
args["w_sm_scales"] = [0.0, 0, 0, 0, 0]

constraints_model_output_path = overfit_w_constraints(
    template_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    unlabeled_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
    unlabeled_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
    two_d_constraints_path="/home/shahar/cardio_corr/outputs/new_runs/outputs_20230724_120732/constraints.npy",#"/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/ex_2d_constraints.npy",
    flows_gt_path='/home/shahar/cardio_corr/outputs/synthetic_dataset23/thetas_90.0_0.0_rs_1.0_1.0_h_1.0/flow_for_image_thetas_90.0_0.0_rs_1.0_1.0_h_1.0.npy',
    args=EasyDict(args)
    )

# args["load"] = get_checkpoints_path(constraints_model_output_path)

# constraints_inference_output_path = infer_w_constraints(
#     template_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
#     unlabeled_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
#     template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
#     unlabeled_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy",
#     two_d_constraints_path="/home/shahar/cardio_corr/outputs/new_runs/outputs_20230725_073457/constraints.npy",#"/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/ex_2d_constraints.npy",
#     save_mask=False,
#     args=EasyDict(args)
#     )

print("completed dummy run")

# future work: 
# save 2d_constraints with model weights so that inference will not require the constraints as an input