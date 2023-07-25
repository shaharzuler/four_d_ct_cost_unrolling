import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils import gradient


# Credit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def TernaryLoss(img, img_warp, max_distance=1): #NOT TESTED
    patch_size = 2 * max_distance + 1

    def _ternary_transform(image):
        out_channels = patch_size * patch_size * patch_size
        w = torch.eye(patch_size) 
        w = w.repeat(out_channels, 1, patch_size, 1, 1)
        w = w.type_as(img)
        patches = F.conv3d(image, w, padding=max_distance)
        transf = patches - image
        transf = transf / torch.sqrt(0.81 + torch.pow(transf, 2))  # norm
        return transf

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist = dist / (0.1 + dist)  # norm
        dist = torch.mean(dist, 1, keepdim=True)  # mean instead of sum
        return dist

    def _valid_mask(t, padding):
        N, C, H, W, D = t.size()
        inner = torch.ones(N, 1, H - 2 * padding, W - 2 * padding, D - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 8)
        return mask

    t1 = _ternary_transform(img)
    t2 = _ternary_transform(img_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(img, max_distance)

    return dist * mask


def smooth_grad_1st(flo:torch.Tensor, vox_dims:torch.Tensor) -> torch.Tensor: 
    weights_x = 1
    weights_y = 1
    weights_z = 1

    dx, dy, dz = gradient(flo, vox_dims)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2.
    loss_z = weights_z * dz.abs() / 2.

    return loss_x.mean() / 3. + loss_y.mean() / 3. + loss_z.mean() / 3.


def SSIM(x:torch.Tensor, y:torch.Tensor, md=1) -> torch.Tensor:
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool3d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool3d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    sigma_x = nn.AvgPool3d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool3d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool3d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist
