import os
import tempfile

from matplotlib import pyplot as plt
import torch
import numpy as np

import three_d_data_manager
from .torch_utils import torch_to_np, mask_xyz_to_13xyz


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, print_precision=3, names=None):
        self.meters = i
        self.print_precision = print_precision
        self.reset(self.meters)
        self.names = names
        if names is not None:
            assert self.meters == len(self.names)
        else:
            self.names = [''] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert (len(val) == self.meters and len(n) == self.meters)
        for i in range(self.meters):
            self.count[i] += n[i]
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def get_avg_meter_name(self, name:str) -> float:
        if self.names is not None:
            try:
                index = self.names.index(name)
                return self.avg[index]
            except AttributeError:
                return None

    def __repr__(self):
        val = ' '.join(['{} {:.{}f}'.format(n, v, self.print_precision) for n, v in
                        zip(self.names, self.val)])
        avg = ' '.join(['{} {:.{}f}'.format(n, a, self.print_precision) for n, a in
                        zip(self.names, self.avg)])
        return '{} ({})'.format(val, avg)


def calc_epe(flows_gt, flows_pred): 
    epe_map = calc_epe_map(flows_gt, flows_pred)
    # epe_map = torch.abs(flow12 - flow12_net).to(self.device).mean()
    error = float(epe_map.mean().item())
    return error, epe_map

def calc_epe_map(flows_gt, flows_pred):
    flow_diff = flows_gt - flows_pred
    epe_map = torch.sqrt(torch.sum(torch.square(flow_diff), dim=1))#.mean()
    return epe_map

def calc_error_in_mask(flows_gt, flows_pred, template_seg):
    if template_seg.sum() == 0:
        return 0.0
    template_seg = mask_xyz_to_13xyz(template_seg).to(flows_gt.device)
    normalization_term = torch.numel(template_seg[0,0]) / template_seg[0,0].nonzero().shape[0]
    error, error_map = (calc_epe(flows_gt * template_seg, flows_pred * template_seg)) 
    error *= normalization_term
    return error, error_map

def calc_error_on_surface(flows_gt, flows_pred, template_seg):
    surface_mask = torch.tensor(three_d_data_manager.extract_segmentation_envelope(torch_to_np(template_seg)))
    surface_mask = mask_xyz_to_13xyz(surface_mask).to(flows_gt.device)
    normalization_term = torch.numel(surface_mask[0,0]) / surface_mask[0,0].nonzero().shape[0]
    error, _ = (calc_epe(flows_gt * surface_mask, flows_pred * surface_mask))
    error *=  normalization_term 
    return error

def calc_measurement_components_on_surface(measurement, surface_mask, surface_normalization_term, voxelized_normals, measurement_locally_radial_denum=None, measurement_locally_tangential_denum=None):
    measurement_locally_radial_size = (measurement * voxelized_normals).sum(1)
    measurement_locally_radial_vectors = measurement_locally_radial_size * voxelized_normals * surface_mask
    if measurement_locally_radial_denum is not None:
        measurement_locally_radial_vectors /= measurement_locally_radial_denum
    measurement_locally_tangential_vectors = (measurement - measurement_locally_radial_vectors) * surface_mask
    if measurement_locally_tangential_denum is not None:
        measurement_locally_tangential_vectors /= measurement_locally_tangential_denum
    measurement_locally_tangential_size = torch.nansum(measurement_locally_tangential_vectors,axis=1)
    
    mean_locally_radial_measurement = abs(float(measurement_locally_radial_size.mean().item()) * surface_normalization_term)
    mean_locally_tangential_measurement = abs(float(measurement_locally_tangential_size.mean().item()) * surface_normalization_term)
    
    return mean_locally_radial_measurement, mean_locally_tangential_measurement, measurement_locally_radial_vectors, measurement_locally_tangential_vectors

def calc_error_vs_distance(flows_pred, flows_gt, distance_validation_masks):
    distance_calculated_errors = {}
    rel_distance_calculated_errors = {}
    for region_name, region in distance_validation_masks.items():
        distance_calculated_errors[region_name] = [[],[]]
        rel_distance_calculated_errors[region_name] = [[],[]]
        for distance, distance_mask in region.items():
            distance_error, error_map = calc_error_in_mask(flows_gt, flows_pred, distance_mask)
            denum_error, gt_map = calc_error_in_mask(flows_gt, torch.zeros_like(flows_pred), distance_mask)
            distance_calculated_errors[region_name][0].append(distance)
            distance_calculated_errors[region_name][1].append(distance_error)
            rel_distance_calculated_errors[region_name][0].append(distance)
            rel_distance_calculated_errors[region_name][1].append(distance_error/denum_error if denum_error>0 else 0)
    return distance_calculated_errors, rel_distance_calculated_errors

def get_error_vs_distance_plot_image(distance_validation_masks, distance_calculated_errors):
    plt.close()
    for region_name, region in distance_calculated_errors.items():
        plt.plot(region[0], region[1])
    plt.xlabel("Distance [pixels]")
    plt.ylabel("Error")
    plt.legend(list(distance_validation_masks.keys()))
    plt.ylim(0,8)
    plt.show()
    ftmp = tempfile.NamedTemporaryFile(suffix='.jpg', prefix='tmp', delete=False)
    ftmp.close()
    plt.savefig(ftmp.name)
    error_vs_dist_plot = np.expand_dims(plt.imread(ftmp.name), 0)
    os.remove(ftmp.name)
    return error_vs_dist_plot


def calc_iou(mask1:np.ndarray, mask2:np.ndarray) -> float: 
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection/union
    return iou