from easydict import EasyDict


from four_d_ct_cost_unrolling import overfit_backbone, infer_backbone, overfit_w_constraints, infer_w_constraints
from four_d_ct_cost_unrolling import get_default_backbone_config, get_default_w_constraints_config, get_checkpoints_path


two_d_constraints_path = r"/home/shahar/cardio_corr/outputs/outputs_20230719_190750/constraints.npy"
template_synthetic_img_path   = r'/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp_onlyz/thetas_0.0_0.0_rs_0.5_0.5_h_1.0/image_skewed_thetas_0.0_0.0_rs_0.5_0.5_h_1.0.npy'#'/home/shahar/cardio_corr/outputs/synthetic_dataset9/thetas_0.0_0.0_rs_0.5_0.5_h_1.0/image_skewed_thetas_0.0_0.0_rs_0.5_0.5_h_1.0.npy'
unlabeled_synthetic_img_path  = r'/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp_onlyz/thetas_0.0_0.0_rs_0.5_0.5_h_1.0/image_orig_thetas_0.0_0.0_rs_0.5_0.5_h_1.0.npy'
template_synthetic_mask_path  = r'/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp_onlyz/thetas_0.0_0.0_rs_0.5_0.5_h_1.0/mask_skewed_thetas_0.0_0.0_rs_0.5_0.5_h_1.0.npy'#'/home/shahar/cardio_corr/outputs/synthetic_dataset9/thetas_0.0_0.0_rs_0.5_0.5_h_1.0/mask_skewed_thetas_0.0_0.0_rs_0.5_0.5_h_1.0.npy'
unlabeled_synthetic_mask_path = r'/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp_onlyz/thetas_0.0_0.0_rs_0.5_0.5_h_1.0/mask_orig_thetas_0.0_0.0_rs_0.5_0.5_h_1.0.npy'
synthetic_flow_path           = r'/home/shahar/cardio_corr/my_packages/cardio_volume_skewer_project/cardio_volume_skewer/outputs/self_validation_params_exp_onlyz/thetas_0.0_0.0_rs_0.5_0.5_h_1.0/flow_for_image_thetas_0.0_0.0_rs_0.5_0.5_h_1.0.npy'#'/home/shahar/cardio_corr/outputs/synthetic_dataset9/thetas_0.0_0.0_rs_0.5_0.5_h_1.0/flow_for_image_thetas_0.0_0.0_rs_0.5_0.5_h_1.0.npy' #  r'/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/onlyx5.npy'#


args = get_default_backbone_config()
args["save_iter"] = 10
args["inference_args"]["inference_flow_median_filter_size"] = False
args["epochs"] = 1000
args["valid_type"] = "synthetic+basic"
args["w_sm_scales"] = [0,0,0,0,0]


backbone_model_output_path = overfit_backbone(
    template_image_path=template_synthetic_img_path,#'/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/image_orig_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',
    unlabeled_image_path=unlabeled_synthetic_img_path,#'/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/image_skewed_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',
    template_seg_path=template_synthetic_mask_path,#'/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/mask_orig_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',
    unlabeled_seg_path=unlabeled_synthetic_mask_path,#'/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/mask_skewed_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',
    flows_gt_path=synthetic_flow_path,#'/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/flow_for_image_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',
    args=EasyDict(args)
    )

    

# args["load"] = get_checkpoints_path(backbone_model_output_path)

# backbone_inference_output_path = infer_backbone(
#     template_image_path='/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/image_orig_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',
#     unlabeled_image_path='/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/image_skewed_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',
#     template_seg_path='/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/mask_orig_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',
#     unlabeled_seg_path='/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/mask_skewed_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',
#     flows_gt_path='/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/flow_for_image_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',
#     args=EasyDict(args)
#     )


# backbone_model_output_path = r"/home/shahar/cardio_corr/outputs/outputs_20230715_164141/outputs_backbone_training_20230715_183029"

# args = get_default_w_constraints_config()
# args["save_iter"] = 2
# args["inference_args"]["inference_flow_median_filter_size"] = False
# args["epochs"] = 300
# args["load"] = get_checkpoints_path(backbone_model_output_path)
# args["valid_type"] = "synthetic+basic"
# args["w_sm_scales"] = [0,0,0,0,0]



# constraints_model_output_path = overfit_w_constraints(
#     template_image_path=template_synthetic_img_path,#'/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/image_orig_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp2/img_orig_thetas_thetas_40_-40_rs_0.5_0.5_h_0.8.nrrd.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/_thetas_40_-40_rs_0.5_0.5_h_0.8_WINNER/img_skewed_thetas_0.0_0.0_rs_1.0_1.0_h_1.0.npy",#"step0.npy", #18 orig i think
#     unlabeled_image_path=unlabeled_synthetic_img_path,#'/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/image_skewed_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp2/img_skewed_thetas_40_-40_rs_0.5_0.5_h_0.8.nrrd.npy",#,"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/_thetas_40_-40_rs_0.5_0.5_h_0.8_WINNER/img_skewed_thetas_16.0_-16.0_rs_0.8_0.8_h_0.92.npy",#"step1.npy", #streched 18 i think
#     template_seg_path=template_synthetic_mask_path,#'/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/mask_orig_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',#"/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy",#"/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/warped_seg_maps/seg_18_to_28.npy",#"/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_raw.npy", #18
#     unlabeled_seg_path=unlabeled_synthetic_mask_path,#'/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/mask_skewed_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',#"/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy",#"/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/warped_seg_maps/seg_18_to_28.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/binary_mask_18.npy",  #28
#     two_d_constraints_path=two_d_constraints_path,#'/home/shahar/cardio_corr/outputs/outputs_20230714_005246/constraints.npy',#"/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/ex_2d_constraints.npy",
#     flows_gt_path=synthetic_flow_path,#'/home/shahar/cardio_corr/outputs/synthetic_dataset2/thetas_40.0_-40.0_rs_0.6_0.6_h_0.8/flow_for_image_thetas_40.0_-40.0_rs_0.6_0.6_h_0.8.npy',#"/home/shahar/cardio_corr/outputs/outputs_20230714_005246/constraints.npy",#/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp2/flow_thetas_40_-40_rs_0.5_0.5_h_0.8.nrrd.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/_thetas_40_-40_rs_0.5_0.5_h_0.8_WINNER/flow_thetas_16.0_-16.0_rs_0.8_0.8_h_0.92.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_70_-70_rs_0.25_0.25_h_0.5.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_40.0_-40.0_rs_0.5_0.5_h_0.8.npy", #flows_gt_path="/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_70_-70_rs_0.25_0.25_h_0.5.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_40.0_-40.0_rs_0.5_0.5_h_0.8.npy", 
#     args=EasyDict(args)
#     )

# args["load"] = get_checkpoints_path(constraints_model_output_path)

# constraints_inference_output_path = infer_w_constraints(
#     template_image_path="step0.npy", #18 orig i think
#     unlabeled_image_path="step1.npy", #streched 18 i think
#     template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", #18
#     unlabeled_seg_path="/home/shahar/projects/pcd_to_mesh/exploration/binary_mask_28.npy",  #28
#     two_d_constraints_path="/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/ex_2d_constraints.npy",
#     save_mask=False,
#     flows_gt_path="/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_70_-70_rs_0.25_0.25_h_0.5.npy",#"/home/shahar/projects/pcd_to_mesh/exploration/self_validation_params_exp/tmp/flow_thetas_40.0_-40.0_rs_0.5_0.5_h_0.8.npy", 
#     args=EasyDict(args)
#     )

# print("completed dummy run")

# future work: 
# save 2d_constraints with model weights so that inference will not require the constraints as an input