from four_d_ct_cost_unrolling import overfit_backbone
# from four_d_ct_cost_unrolling.four_d_ct_cost_unrolling.src.main import overfit_backbone


args = {
    "search_range": 4,
    "admm_args.rho": 0.1,
    "admm_args.lamb": 0.1,
    "admm_args.eta": 1,
    "admm_args.grad": "1st", ##
    "admm_args.T": 1,
    "admm_args.alpha": 50,
    "admm_args.learn_mask": False,
    "admm_args.apply_admm": [0,0,0,0,1],
    "w_admm": [1.0, 0.0, 0.0, 0.0, 0.0],
    "admm_rho": 0.1,
    "w_l1": 1,
    "w_ssim": 1,
    "w_ternary": 1,
    "alpha": 10,
    "w_scales": [3.0, 3.0, 3.0, 3.0, 3.0],
    "args.w_constraints_scales": [1,1,1,1,1],
    "loss": 'unflow+2d_constraints',
    "epoch_size": 100,
    "plot_freq": 1,
    "load": 'outputs/checkpoints/overfitting_from_28/25_unflowloss/l2r_4dct_costunrolling__cyc_ckpt.pth.tar',
    "lr": 0.0001,
    "save_root": "...",
    "after_epoch":0,
    ".model_suffix": 'l2r_4dct_costunrolling__cyc',
    "levels": [5000,5000,5000],
    "max_reduce_loss_delay": 10
}

overfit_backbone(
    template_image_path="/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    unlabeled_image_path="/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
    template_seg_path="/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_raw.npy", 
    unlabeled_seg_path="/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_raw.npy", 
    output_ckpts_path="/temp_ckpts", 
    cfg=args
    )