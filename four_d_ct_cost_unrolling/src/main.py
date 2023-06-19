from .losses.get_loss import get_loss

from .trainers.pull_seg_train_framework import PullSegmentationMapTrainFramework, PullSegmentationMapTrainFrameworkInference
from .trainers.pull_seg_train_framework_w_2d_constraints import PullSegmentationMapTrainFrameworkWith2dConstraints

from .dataset_handlers.seg_puller_cardio_ct_dataset import SegmentationPullerCardioDataset, SegmentationPullerCardioDatasetWithConstraints, SegmentationPullerCardioSampleArgs, SegmentationPullerCardiosampleWithConstraintsArgs

from .models.pwc3d import PWC3D
from .models.pwc3d_w_2d_constraints import PWC3Dw2dConstraints

# l2r_costunrolling_shahar_warp_seg_masks_unflowloss.json
# l2r_costunrolling_shahar_warp_seg_masks_unflowloss_2dconstraints_l_included_not_weighted.json
# l2r_costunrolling_shahar_warp_seg_masks_infer.json

# args = {
#     "search_range": 4,
#     "admm_args.rho": 0.1,
#     "admm_args.lamb": 0.1,
#     "admm_args.eta": 1,
#     "admm_args.grad": "1st", ##
#     "admm_args.T": 1,
#     "admm_args.alpha": 50,
#     "admm_args.learn_mask": False,
#     "admm_args.apply_admm": [0,0,0,0,1],
#     "w_admm": [1.0, 0.0, 0.0, 0.0, 0.0],
#     "admm_rho": 0.1,
#     "w_l1": 1,
#     "w_ssim": 1,
#     "w_ternary": 1,
#     "alpha": 10,
#     "w_scales": [3.0, 3.0, 3.0, 3.0, 3.0],
#     "args.w_constraints_scales": [1,1,1,1,1],
#     "loss": 'unflow+2d_constraints',
#     "epoch_size": 100,
#     "plot_freq": 1,
#     "load": 'outputs/checkpoints/overfitting_from_28/25_unflowloss/l2r_4dct_costunrolling__cyc_ckpt.pth.tar',
#     "lr": 0.0001,
#     "save_root": "...",
#     "after_epoch":0,
#     ".model_suffix": 'l2r_4dct_costunrolling__cyc',
#     "levels": [5000,5000,5000],
#     "max_reduce_loss_delay": 10
# }

    

def overfit_backbone(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path, output_ckpts_path, args=None):
    data_sample_args = SegmentationPullerCardioSampleArgs(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path)
    train_set = SegmentationPullerCardioDataset(data_sample_args)
    model = PWC3D(args)
    loss = {"loss_module" : get_loss(args)}
    trainer = PullSegmentationMapTrainFramework(train_set, model, loss, args)
    trainer.train(0, 1) 

def infer_backbone(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path):
    data_sample_args = SegmentationPullerCardioSampleArgs(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path)
    train_set = SegmentationPullerCardioDataset(data_sample_args)
    model = PWC3D(args)
    loss = {"loss_module" : get_loss(args)}
    trainer = PullSegmentationMapTrainFrameworkInference(train_set, model, loss, args)
    trainer.train(0, 1)

def overfit_w_constraints(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path, two_d_constraints_path):
    data_sample_args = SegmentationPullerCardiosampleWithConstraintsArgs(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path, two_d_constraints_path)
    train_set = SegmentationPullerCardioDatasetWithConstraints(data_sample_args)
    model = PWC3Dw2dConstraints(args)
    loss = {"loss_module" : get_loss(args)}
    trainer = PullSegmentationMapTrainFrameworkWith2dConstraints(train_set, model, loss, args)
    trainer.train(0, 1)

def infer_w_constraints(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path, two_d_constraints_path):
    data_sample_args = SegmentationPullerCardiosampleWithConstraintsArgs(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path, two_d_constraints_path)
    train_set = SegmentationPullerCardioDatasetWithConstraints(data_sample_args)
    model = PWC3Dw2dConstraints(args)
    loss = {"loss_module" : get_loss(args)}
    trainer = PullSegmentationMapTrainFrameworkWith2dConstraints(train_set, model, loss, args)
    trainer.train(0, 1)

# overfit_backbone(
#     template_image_path="/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
#     unlabeled_image_path="/data/cardiac_3d_data/18/orig/voxels/xyz_arr_raw.npy", 
#     template_seg_path="/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_raw.npy", 
#     unlabeled_seg_path="/data/cardiac_3d_data/18/orig/voxels/xyz_voxels_mask_raw.npy", 
#     output_ckpts_path="/temp_ckpts", 
#     cfg=args
#     )