import torch

def gradient(data:torch.tensor, vox_dims:torch.tensor=torch.tensor((1, 1, 1))) -> tuple[torch.tensor]:
    D_dy = (data[:, :, 1:] - data[:, :, :-1])/vox_dims[1]
    D_dx = (data[:, :, :, 1:] - data[:, :, :, :-1])/vox_dims[0]
    D_dz = (data[:, :, :, :, 1:] - data[:, :, :, :, :-1])/vox_dims[2]

    return D_dx, D_dy, D_dz



