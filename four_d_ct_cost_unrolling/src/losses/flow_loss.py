from math import exp
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_utils import gradient
# from .utils.misc import log



# Credit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def TernaryLoss(img, img_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _ternary_transform(image):
        # log(f'Image size={image.size()}')

        out_channels = patch_size * patch_size * patch_size

        w = torch.eye(patch_size) 
        w = w.repeat(out_channels, 1, patch_size, 1, 1)
        # log(f'size={w.size()}')

        w = w.type_as(img)
        # log(f'weights size={w.size()}')
        patches = F.conv3d(image, w, padding=max_distance)
        # log(f'patches size={patches.size()}')
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
        inner = torch.ones(N, 1, H - 2 * padding, W - 2 *
                           padding, D - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 8)
        return mask

    t1 = _ternary_transform(img)
    t2 = _ternary_transform(img_warp)
    dist = _hamming_distance(t1, t2)
    # log(f'dist size={dist.size()}')
    mask = _valid_mask(img, max_distance)
    # log(f'mask size={mask.size()}')

    return dist * mask


def smooth_grad_1st(flo, image, vox_dims, alpha, flow_only=True, seg_recons=None):
    weights_x = 1
    weights_y = 1
    weights_z = 1
    if not flow_only:
        img_dx, img_dy, img_dz = gradient(image, vox_dims)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx),
                                          1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy),
                                          1, keepdim=True) * alpha)
        weights_z = torch.exp(-torch.mean(torch.abs(img_dz),
                                          1, keepdim=True) * alpha)

    dx, dy, dz = gradient(flo, vox_dims)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2.
    loss_z = weights_z * dz.abs() / 2.

    if seg_recons is not None:
        _, _, h, w, d = loss_x.shape
        loss_x[:,0,:,:,:] = loss_x[:,0,:,:,:]*seg_recons[:,:,:h,:w,:d]
        loss_x[:,1,:,:,:] = loss_x[:,1,:,:,:]*seg_recons[:,:,:h,:w,:d]
        loss_x[:,2,:,:,:] = loss_x[:,2,:,:,:]*seg_recons[:,:,:h,:w,:d]
        _, _, h, w, d = loss_y.shape
        loss_y[:,0,:,:,:] = loss_y[:,0,:,:,:]*seg_recons[:,:,:h,:w,:d]
        loss_y[:,1,:,:,:] = loss_y[:,1,:,:,:]*seg_recons[:,:,:h,:w,:d]
        loss_y[:,2,:,:,:] = loss_y[:,2,:,:,:]*seg_recons[:,:,:h,:w,:d]
        _, _, h, w, d = loss_z.shape
        loss_z[:,0,:,:,:] = loss_z[:,0,:,:,:]*seg_recons[:,:,:h,:w,:d]
        loss_z[:,1,:,:,:] = loss_z[:,1,:,:,:]*seg_recons[:,:,:h,:w,:d]
        loss_z[:,2,:,:,:] = loss_z[:,2,:,:,:]*seg_recons[:,:,:h,:w,:d]


    return loss_x.mean() / 3. + loss_y.mean() / 3. + loss_z.mean() / 3.



def gaussian(window_size, sigma):
    print(f'gaussian 1')
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    print(f'gaussian 2')
    return gauss/gauss.sum()


def create_window_3D(window_size, channel):
    print(f'create_window_3D 1')
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(
        1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(
        channel, 1, window_size, window_size, window_size).contiguous())
    print(f'create_window_3D 2')
    return window


def SSIM(x, y, md=1):
    log(f'Running SSIM with x={x}, y={y}')
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
