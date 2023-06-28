from .losses.get_loss import get_loss

from .trainers.pull_seg_train_framework import PullSegmentationMapTrainFramework
from .trainers.pull_seg_train_framework_w_2d_constraints import PullSegmentationMapTrainFrameworkWith2dConstraints

from .dataset_handlers.seg_puller_cardio_ct_dataset import SegmentationPullerCardioDataset, SegmentationPullerCardioDatasetWithConstraints, SegmentationPullerCardioSampleArgs, SegmentationPullerCardiosampleWithConstraintsArgs, SegmentationPullerSample

from .models.pwc3d import PWC3D
from .models.pwc3d_w_2d_constraints import PWC3Dw2dConstraints

# l2r_costunrolling_shahar_warp_seg_masks_unflowloss.json
# l2r_costunrolling_shahar_warp_seg_masks_unflowloss_2dconstraints_l_included_not_weighted.json
# l2r_costunrolling_shahar_warp_seg_masks_infer.json


    

def overfit_backbone(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path, output_ckpts_path, args=None): #todo output path
    data_sample_args = SegmentationPullerCardioSampleArgs(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path)
    train_set = SegmentationPullerCardioDataset(data_sample_args, sample_type=SegmentationPullerSample)
    model = PWC3D(args)
    loss = get_loss(args)
    trainer = PullSegmentationMapTrainFramework(train_set, model, loss, args)
    trainer.train(0, 1) 

def infer_backbone(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path, args=None):
    data_sample_args = SegmentationPullerCardioSampleArgs(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path)
    train_set = SegmentationPullerCardioDataset(data_sample_args, sample_type=SegmentationPullerSample)
    model = PWC3D(args)
    loss = get_loss(args) # is loss nesseccary? TODO
    trainer = PullSegmentationMapTrainFramework(train_set, model, loss, args)
    trainer.infer(0, 1)

def overfit_w_constraints(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path, two_d_constraints_path, args=None):
    data_sample_args = SegmentationPullerCardiosampleWithConstraintsArgs(
        template_image_path=template_image_path, 
        unlabeled_image_path=unlabeled_image_path, 
        template_seg_path=template_seg_path, 
        unlabeled_seg_path=unlabeled_seg_path, 
        two_d_constraints_path=two_d_constraints_path)
    train_set = SegmentationPullerCardioDatasetWithConstraints(data_sample_args)
    model = PWC3Dw2dConstraints(args)
    loss = get_loss(args)
    trainer = PullSegmentationMapTrainFrameworkWith2dConstraints(train_set, model, loss, args)
    trainer.train(0, 1)

def infer_w_constraints(template_image_path, unlabeled_image_path, template_seg_path, unlabeled_seg_path, save_mask, two_d_constraints_path, args=None):
    data_sample_args = SegmentationPullerCardiosampleWithConstraintsArgs(template_image_path=template_image_path, 
        unlabeled_image_path=unlabeled_image_path, 
        template_seg_path=template_seg_path, 
        unlabeled_seg_path=unlabeled_seg_path, 
        two_d_constraints_path=two_d_constraints_path)
    train_set = SegmentationPullerCardioDatasetWithConstraints(data_sample_args)
    model = PWC3Dw2dConstraints(args)
    loss = get_loss(args) # is loss nesseccary? TODO
    trainer = PullSegmentationMapTrainFrameworkWith2dConstraints(train_set, model, loss, args)
    trainer.infer(0, 1, save_mask=save_mask)

