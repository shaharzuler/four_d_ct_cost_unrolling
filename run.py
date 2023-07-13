from easydict import EasyDict


from four_d_ct_cost_unrolling import overfit_backbone, infer_backbone, overfit_w_constraints, infer_w_constraints
from four_d_ct_cost_unrolling import get_default_backbone_config, get_default_w_constraints_config, get_checkpoints_path


args = get_default_backbone_config()
args["save_iter"] = 2
args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 10

backbone_model_output_path = overfit_backbone(
    template_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    unlabeled_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
    unlabeled_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
    flows_gt_path='/home/shahar/cardio_corr/outputs/synthetic_dataset/thetas_40.0_-40.0_rs_0.5_0.5_h_0.8/flow_thetas_40.0_-40.0_rs_0.5_0.5_h_0.8.npy', 
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



args = get_default_w_constraints_config()
args["save_iter"] = 2
args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 10
args["load"] = get_checkpoints_path(backbone_model_output_path)


constraints_model_output_path = overfit_w_constraints(
    template_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    unlabeled_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
    unlabeled_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
    two_d_constraints_path="/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/ex_2d_constraints.npy",
    flows_gt_path='/home/shahar/cardio_corr/outputs/synthetic_dataset/thetas_40.0_-40.0_rs_0.5_0.5_h_0.8/flow_thetas_40.0_-40.0_rs_0.5_0.5_h_0.8.npy',
    args=EasyDict(args)
    )

args["load"] = get_checkpoints_path(constraints_model_output_path)

constraints_inference_output_path = infer_w_constraints(
    template_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    unlabeled_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
    unlabeled_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy",
    two_d_constraints_path="/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/ex_2d_constraints.npy",
    save_mask=False,
    args=EasyDict(args)
    )

print("completed dummy run")

# future work: 
# save 2d_constraints with model weights so that inference will not require the constraints as an input