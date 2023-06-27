from typing import Union
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

def _mesh_grid(B, H, W, D):
    # batches not implented
    x = torch.arange(H)
    y = torch.arange(W)
    z = torch.arange(D)
    mesh = torch.stack(torch.meshgrid(x, y, z)[::-1], 0)
    mesh = mesh.unsqueeze(0)
    return mesh.repeat([B,1,1,1,1])

def _norm_grid(v_grid):
    _, _, H, W, D = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (D - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 2, :, :] = 2.0 * v_grid[:, 2, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 4, 1)

def flow_warp(img2, flow12, pad='border', mode='bilinear'):
    assert (img2.shape[-3:] == flow12.shape[-3:])
    B, _, H, W, D = flow12.size()
    flow12 = torch.flip(flow12, [1])
    base_grid = _mesh_grid(B, H, W, D).type_as(img2)  # B2HW

    v_grid = _norm_grid(base_grid + flow12)  # BHW2
    im1_recons = nn.functional.grid_sample(img2, v_grid, mode=mode, padding_mode=pad, align_corners=True)

    return im1_recons

def extract_segmentation_envelope(seg_arr):
    return ((np.abs(np.logical_or(*np.gradient(seg_arr.astype(int),0.5))))*seg_arr).astype(bool) #TODO move to 3d manager

def _get_constraints_closest_indices(constraints_indices, envelope_indices):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(constraints_indices)
    nn_ind = neigh.kneighbors(envelope_indices, return_distance=False) # shape (N2, 1)
    closest_constraints_indices = constraints_indices[nn_ind][:,0,:] # shape (N2, 3)
    return closest_constraints_indices

def _restore_constraints(constraints_arr, envelope_indices, closest_constraints_indices):
    restored_constraints_arr = np.empty(constraints_arr.shape)
    restored_constraints_arr[:] = np.nan
    for axis in range(3):
        restored_constraints_arr[envelope_indices[:,0],envelope_indices[:,1],envelope_indices[:,2],axis] = constraints_arr[closest_constraints_indices[:,0],closest_constraints_indices[:,1],closest_constraints_indices[:,2],axis]
    return restored_constraints_arr

def attach_flow_between_segs(constraints_arr:np.array, seg_arr:np.array) -> np.array:
    """
    Takes 2d constraints based on one segmentation map (for example a smooth seg map),
    and moves it to each index closest neighbour on the second segmentation map.
    """
    #TODO validate visually!
    envelope = extract_segmentation_envelope(seg_arr)
    constraints_indices = np.array([*np.where(constraints_arr[:,:,:,0] != np.nan)]).T # shape N1,3
    envelope_indices = np.array([*np.where(envelope)]).T # shape N2,3
    closest_constraints_indices = _get_constraints_closest_indices(constraints_indices, envelope_indices)
    restored_constraints_arr = _restore_constraints(constraints_arr, envelope_indices, closest_constraints_indices)
    return restored_constraints_arr

def rescale_flow_tensor(flow_tensor:torch.tensor, target_shape:tuple[int,int,int,int,int], mode:str="area", return_scale_factor=False) -> Union[torch.tensor,tuple[torch.tensor, tuple]]:
    """
    flow is a 5D arr shape n,3,x,y,z
    """
    scale_factor = np.array(target_shape[2:])/np.array(flow_tensor.shape[2:])
    if flow_tensor.shape != target_shape:
        scale_factor = np.array(target_shape[2:])/np.array(flow_tensor.shape[2:])
        for i in range(3):
            flow_tensor[:,i,:,:,:] *= scale_factor[i]
        flow_tensor = F.interpolate(flow_tensor, scale_factor=tuple(scale_factor), mode=mode)
    if return_scale_factor:
        return flow_tensor, tuple(scale_factor)
    return flow_tensor

def xyz3_to_3xyz(flow:np.array) -> np.array:
    return np.transpose(flow, (3,0,1,2))