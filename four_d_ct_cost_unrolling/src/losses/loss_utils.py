from typing import Tuple
import torch

def gradient(data:torch.Tensor, vox_dims:torch.Tensor=torch.Tensor((1, 1, 1))) -> Tuple[torch.Tensor]:
    D_dy = (data[:, :, 1:] - data[:, :, :-1])/vox_dims[1]
    D_dx = (data[:, :, :, 1:] - data[:, :, :, :-1])/vox_dims[0]
    D_dz = (data[:, :, :, :, 1:] - data[:, :, :, :, :-1])/vox_dims[2]

    return D_dx, D_dy, D_dz



