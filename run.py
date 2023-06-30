from easydict import EasyDict

from four_d_ct_cost_unrolling import overfit_backbone, infer_backbone, overfit_w_constraints, infer_w_constraints


args = {
    "search_range": 4,
    "admm_args": {
        "rho": 0.1,
        "lamb": 0.1,
        "eta": 1,
        "grad": "1st", 
        "T": 1,
        "alpha": 50,
        "learn_mask": False,
        "apply_admm": [0,0,0,0,1],
    },
    "w_admm": [1.0, 0.0, 0.0, 0.0, 0.0],
    "admm_rho": 0.1,
    "w_l1": 1,
    "w_ssim": 1,
    "w_ternary": 1,
    "alpha": 10,
    "w_scales": [3.0, 3.0, 3.0, 3.0, 3.0],
    "args.w_constraints_scales": [1,1,1,1,1],
    "loss": 'unflow',
    "plot_freq": 1,
    "load": '/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/four_d_ct_cost_unrolling/four_d_ct_cost_unrolling/src/checkpoints/4DCT_best_w_admm_ckpt.pth.tar', #TODO make relative path after i move this args to a defaults args section
    "lr": 0.0001,
    "output_root": "outputs_backbone_training",
    "after_epoch":0,
    "model_suffix": '4dct_costunrolling',
    "epochs": 3, #TODO 5000,
    "max_reduce_loss_delay": 10,
    "n_gpu": 2,
    "batch_size": 1,
    "w_sm_scales":[20,0,0,0,0],
    "record_freq": 1,
    "valid_type": "basic",
    "save_iter": 50,
    "inference_args": {
        "inference_flow_median_filter_size": False, #7 TODO update
        "template_timestep": 18,
        "unlabeled_timestep": 28,
        "output_warped_seg_maps_dir": "warped_seg_maps",
    }
    
}

# overfit_backbone(
#     template_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
#     unlabeled_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
#     template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
#     unlabeled_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
#     output_ckpts_path=None, 
#     args=EasyDict(args)
#     )


# infer_backbone(
#     template_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
#     unlabeled_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
#     template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
#     unlabeled_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
#     args=EasyDict(args)
#     )

# TODO mpve mask from cuda

args = {
    "search_range": 4,
    "admm_args": {
        "rho": 0.1,
        "lamb": 0.1,
        "eta": 1,
        "grad": "1st", 
        "T": 1,
        "alpha": 50,
        "learn_mask": False,
        "apply_admm": [0,0,0,0,1],
    },
    "w_admm": [1.0, 0.0, 0.0, 0.0, 0.0],
    "admm_rho": 0.1,
    "w_l1": 1,
    "w_ssim": 1,
    "w_ternary": 1,
    "alpha": 10,
    "w_scales": [3.0, 3.0, 3.0, 3.0, 3.0],
    "args.w_constraints_scales": [1, 1, 1, 1, 1],
    "loss": 'unflow+2d_constraints',
    "plot_freq": 1,
    "load": '/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/four_d_ct_cost_unrolling/four_d_ct_cost_unrolling/src/checkpoints/4DCT_best_w_admm_ckpt.pth.tar', #TODO make relative path after i move this args to a defaults args section
    "lr": 0.0001,
    "output_root": "outputs_constraints_training",
    "after_epoch":0,
    "model_suffix": '4dct_costunrolling',
    "epochs": 80, #TODO 5000,
    "max_reduce_loss_delay": 10,
    "n_gpu": 2,
    "batch_size": 1,
    "w_sm_scales":[20, 0, 0, 0, 0],
    "record_freq": 1,
    "valid_type": "basic",
    "save_iter": 50,
    "w_constraints_scales": [1.0, 1.0, 1.0, 1.0, 1.0],
    "inference_args": {
        "inference_flow_median_filter_size": False, #7 TODO update
        "template_timestep": 18,
        "unlabeled_timestep": 28,
        "output_warped_seg_maps_dir": "warped_seg_maps",
    }
}

# overfit_w_constraints(
#     template_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
#     unlabeled_image_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
#     template_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
#     unlabeled_seg_path="/home/shahar/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_smooth.npy", 
#     two_d_constraints_path="/home/shahar/cardio_corr/my_packages/four_d_ct_cost_unrolling_project/ex_2d_constraints.npy",
#     args=EasyDict(args)
#     )

infer_w_constraints(
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