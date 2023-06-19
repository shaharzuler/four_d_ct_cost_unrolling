import torch




def gradient(data, vox_dims=torch.tensor((1, 1, 1))):
    if len(vox_dims.shape) > 1:
        batch = True
        batch_size = vox_dims.shape[0]
    else:
        batch = False

    if not batch:
        D_dy = (data[:, :, 1:] - data[:, :, :-1])/vox_dims[1]
        D_dx = (data[:, :, :, 1:] - data[:, :, :, :-1])/vox_dims[0]
        D_dz = (data[:, :, :, :, 1:] - data[:, :, :, :, :-1])/vox_dims[2]
    else:
        D_dy = (data[:, :, 1:] - data[:, :, :-1])
        D_dx = (data[:, :, :, 1:] - data[:, :, :, :-1])
        D_dz = (data[:, :, :, :, 1:] - data[:, :, :, :, :-1])
        for sample in range(batch_size):
            # print(f"data:{data.shape}, voxdims:{vox_dims.shape}")
            D_dy[sample] = D_dy[sample]/vox_dims[sample, 1]
            D_dx[sample] = D_dx[sample]/vox_dims[sample, 0]
            D_dz[sample] = D_dz[sample]/vox_dims[sample, 2]

    return D_dx, D_dy, D_dz



