from easydict import EasyDict


from four_d_ct_cost_unrolling import overfit_backbone, infer_backbone, overfit_w_constraints, infer_w_constraints
from four_d_ct_cost_unrolling import get_default_backbone_config, get_default_w_constraints_config, get_checkpoints_path

#############################################################
import nrrd
import numpy as np

# step_0_path = "/home/shahar/projects/pcd_to_mesh/exploration/lv_gdl/self_validation_vids/LV_pixels_move/r0.75_h0.9_theta40/img_skewed_thetas_0.0_0.0_rs_1.0_1.0_h_1.0.nrrd"
# step_1_path = "/home/shahar/projects/pcd_to_mesh/exploration/lv_gdl/self_validation_vids/LV_pixels_move/r0.75_h0.9_theta40/img_skewed_thetas_40.0_-40.0_rs_0.75_0.75_h_0.9.nrrd"
# step_1_flow_path = "/home/shahar/projects/pcd_to_mesh/exploration/lv_gdl/self_validation_vids/LV_pixels_move/r0.75_h0.9_theta40/flow_thetas_40.0_-40.0_rs_0.75_0.75_h_0.9.nrrd"


# step_0_path = "/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/_thetas_40_-40_rs_0.5_0.5_h_0.8_WINNER/img_skewed_thetas_0.0_0.0_rs_1.0_1.0_h_1.0.nrrd"
# step_1_path = "/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/_thetas_40_-40_rs_0.5_0.5_h_0.8_WINNER/img_skewed_thetas_16.0_-16.0_rs_0.8_0.8_h_0.92.nrrd"
# step_1_flow_path = "/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/_thetas_40_-40_rs_0.5_0.5_h_0.8_WINNER/flow_thetas_16.0_-16.0_rs_0.8_0.8_h_0.92.nrrd"

# step0 = nrrd.read(step_0_path)[0]
# step1 = nrrd.read(step_1_path)[0]
# flow1 = nrrd.read(step_1_flow_path)[0]

# np.save(step_0_path.replace(".nrrd", ".npy"), step0)
# np.save(step_1_path.replace(".nrrd", ".npy"), step1)
# np.save(step_1_flow_path.replace(".nrrd", ".npy"), flow1)

# print(1)
# #############################################################


args = get_default_backbone_config()
args["save_iter"] = 2
args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 1000
args["valid_type"] = "synthetic+basic"
args["w_sm_scales"] = [0,0,0,0,0]

"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp2/img_skewed_thetas_40_-40_rs_0.5_0.5_h_0.8.nrrd.npy"
"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp2/img_orig_thetas_thetas_40_-40_rs_0.5_0.5_h_0.8.nrrd.npy"
"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp2/flow_thetas_40_-40_rs_0.5_0.5_h_0.8.nrrd.npy"



backbone_model_output_path = overfit_backbone(
    template_image_path="/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp2/img_orig_thetas_thetas_40_-40_rs_0.5_0.5_h_0.8.nrrd.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/_thetas_40_-40_rs_0.5_0.5_h_0.8_WINNER/img_skewed_thetas_0.0_0.0_rs_1.0_1.0_h_1.0.npy",#"step0.npy", #18 orig i think
    unlabeled_image_path="/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp2/img_skewed_thetas_40_-40_rs_0.5_0.5_h_0.8.nrrd.npy",#,"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/_thetas_40_-40_rs_0.5_0.5_h_0.8_WINNER/img_skewed_thetas_16.0_-16.0_rs_0.8_0.8_h_0.92.npy",#"step1.npy", #streched 18 i think
    template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy",#"/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/warped_seg_maps/seg_18_to_28.npy",#"/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_raw.npy", #18
    unlabeled_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy",#"/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/warped_seg_maps/seg_18_to_28.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/binary_mask_18.npy",  #28
    flows_gt_path="/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp2/flow_thetas_40_-40_rs_0.5_0.5_h_0.8.nrrd.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/_thetas_40_-40_rs_0.5_0.5_h_0.8_WINNER/flow_thetas_16.0_-16.0_rs_0.8_0.8_h_0.92.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_70_-70_rs_0.25_0.25_h_0.5.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_40.0_-40.0_rs_0.5_0.5_h_0.8.npy", 
    args=EasyDict(args)
    )

    

args["load"] = get_checkpoints_path(backbone_model_output_path)

backbone_inference_output_path = infer_backbone(
    template_image_path="step0.npy", #18 orig i think
    unlabeled_image_path="step1.npy", #streched 18 i think
    template_seg_path="/home/shahar/projects/pcd_to_mesh/exploration/binary_mask_28.npy",#"/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", #18
    unlabeled_seg_path="/home/shahar/projects/pcd_to_mesh/exploration/binary_mask_28.npy",  #28
    flows_gt_path="/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_70_-70_rs_0.25_0.25_h_0.5.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_40.0_-40.0_rs_0.5_0.5_h_0.8.npy", 
    args=EasyDict(args)
    )



args = get_default_w_constraints_config()
args["save_iter"] = 2
args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 300
args["load"] = get_checkpoints_path(backbone_model_output_path)
args["valid_type"] = "synthetic+basic"
args["w_sm_scales"] = [0,0,0,0,0]


constraints_model_output_path = overfit_w_constraints(
    template_image_path="step0.npy", #18 orig i think
    unlabeled_image_path="step1.npy", #streched 18 i think
    template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", #18
    unlabeled_seg_path="/home/shahar/projects/pcd_to_mesh/exploration/binary_mask_28.npy",  #28
    two_d_constraints_path="/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/ex_2d_constraints.npy",
    flows_gt_path="/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_70_-70_rs_0.25_0.25_h_0.5.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_40.0_-40.0_rs_0.5_0.5_h_0.8.npy", 
    args=EasyDict(args)
    )

args["load"] = get_checkpoints_path(constraints_model_output_path)

constraints_inference_output_path = infer_w_constraints(
    template_image_path="step0.npy", #18 orig i think
    unlabeled_image_path="step1.npy", #streched 18 i think
    template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", #18
    unlabeled_seg_path="/home/shahar/projects/pcd_to_mesh/exploration/binary_mask_28.npy",  #28
    two_d_constraints_path="/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/ex_2d_constraints.npy",
    save_mask=False,
    flows_gt_path="/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_70_-70_rs_0.25_0.25_h_0.5.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_40.0_-40.0_rs_0.5_0.5_h_0.8.npy", 
    args=EasyDict(args)
    )

print("completed dummy run")

# future work: 
# save 2d_constraints with model weights so that inference will not require the constraints as an input