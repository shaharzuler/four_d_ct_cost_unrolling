import numpy as np
import matplotlib.pyplot as plt
import torch
import flow_vis
# from .flow_utils import resize_flow_tensor #, flow_warp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure
import torch.nn.functional as F
import matplotlib.cm as cm
from scipy.ndimage.interpolation import map_coordinates
import nrrd
import os
import cv2




# def plot_image(data, axes=None, output_path=None, show=True):
#     fig = None
#     if axes is None:
#         fig, axes = plt.subplots(1, 3, figsize=(10, 10))
#     while len(data.shape) > 3:
#         data = data.squeeze(0)
#     indices = np.array(data.shape) // 2
#     i, j, k = indices
#     slice_x = rotate(data[i, :, :])
#     slice_y = rotate(data[:, j, :])
#     slice_z = rotate(data[:, :, k])

#     kwargs = {}
#     kwargs['cmap'] = 'YlGnBu'
#     x_extent, y_extent, z_extent = [(0, b - 1) for b in data.shape]
#     f0 = axes[0].imshow(slice_x, extent=y_extent + z_extent, **kwargs)
#     f1 = axes[1].imshow(slice_y, extent=x_extent + z_extent, **kwargs)
#     f2 = axes[2].imshow(slice_z, extent=x_extent + y_extent, **kwargs)
#     plt.colorbar(f0, ax=axes[0])
#     plt.colorbar(f1, ax=axes[1])
#     plt.colorbar(f2, ax=axes[2])
#     plt.tight_layout()
#     if output_path is not None and fig is not None:
#         fig.savefig(output_path)
#     if show:
#         plt.show()
#     return fig

# def plot_images(
#         img1, img2, img3,
#         axes=None,
#         output_path=None,
#         show=True,):
#     fig = None
#     if axes is None:
#         fig, axes = plt.subplots(3, 3, figsize=(10, 10))
#     while len(img1.shape) > 3:
#         img1 = img1.squeeze(0)
#     while len(img2.shape) > 3:
#         img2 = img2.squeeze(0)
#     while len(img3.shape) > 3:
#         img3 = img3.squeeze(0)

#     indices = np.array(img1.shape) // 2
#     i, j, k = indices
#     slice_x_1 = rotate(img1[i, :, :])
#     slice_y_1 = rotate(img1[:, j, :])
#     slice_z_1 = rotate(img1[:, :, k])
#     slice_x_2 = rotate(img2[i, :, :])
#     slice_y_2 = rotate(img2[:, j, :])
#     slice_z_2 = rotate(img2[:, :, k])
#     slice_x_3 = rotate(img3[i, :, :])
#     slice_y_3 = rotate(img3[:, j, :])
#     slice_z_3 = rotate(img3[:, :, k])
#     kwargs = {}
#     kwargs['cmap'] = 'gray'
#     x_extent, y_extent, z_extent = [(0, b - 1) for b in img1.shape]
#     axes[0][0].imshow(slice_x_1, extent=y_extent + z_extent, **kwargs)
#     axes[0][1].imshow(slice_y_1, extent=x_extent + z_extent, **kwargs)
#     axes[0][2].imshow(slice_z_1, extent=x_extent + y_extent, **kwargs)
#     axes[1][0].imshow(slice_x_2, extent=y_extent + z_extent, **kwargs)
#     axes[1][1].imshow(slice_y_2, extent=x_extent + z_extent, **kwargs)
#     axes[1][2].imshow(slice_z_2, extent=x_extent + y_extent, **kwargs)
#     axes[2][0].imshow(slice_x_3, extent=y_extent + z_extent, **kwargs)
#     axes[2][1].imshow(slice_y_3, extent=x_extent + z_extent, **kwargs)
#     axes[2][2].imshow(slice_z_3, extent=x_extent + y_extent, **kwargs)
#     plt.tight_layout()
#     if output_path is not None and fig is not None:
#         fig.savefig(output_path)
#     if show:
#         plt.show()
#     return fig

# def plot_flow(flow,
#               axes=None,
#               output_path=None,
#               show=True, ):
#     fig = None
#     if axes is None:
#         fig, axes = plt.subplots(1, 3, figsize=(10, 10))
#     while len(flow.shape) > 4:
#         flow = flow.squeeze(0)
#     indices = np.array(flow.shape[1:]) // 2
#     i, j, k = indices

#     slice_x_flow = (flow[1:3, i, :, :])
#     slice_x_flow_col = rotate(flow_vis.flow_to_color(
#         slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
#     slice_y_flow_col = rotate(flow_vis.flow_to_color(
#         slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     slice_z_flow = (flow[0:2, :, :, k])
#     slice_z_flow_col = rotate(flow_vis.flow_to_color(
#         slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     # xy_grid = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
#     # xz_grid = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[2]))
#     kwargs = {}
#     # kwargs['cmap'] = 'gray'
#     x_extent, y_extent, z_extent = [(0, b - 1) for b in flow.shape[1:]]
#     axes[0].imshow(slice_x_flow_col, extent=y_extent + z_extent, **kwargs)
#     axes[1].imshow(slice_y_flow_col, extent=x_extent + z_extent, **kwargs)
#     axes[2].imshow(slice_z_flow_col, extent=x_extent + y_extent, **kwargs)
#     plt.tight_layout()

#     if output_path is not None and fig is not None:
#         fig.savefig(output_path)
#     if show:
#         plt.show()
#     # return slice_x_flow_col, slice_y_flow_col, slice_z_flow_col
#     return fig

# def plot_training_fig(img1, img2, flow,
#                       axes=None,
#                       output_path=None,
#                       show=True, ):
#     fig = None
#     if axes is None:
#         fig, axes = plt.subplots(3, 3, figsize=(10, 10))

#     while len(img1.shape) > 3:
#         img1 = img1.squeeze(0)
#     while len(img2.shape) > 3:
#         img2 = img2.squeeze(0)
#     while len(flow.shape) > 4:
#         flow = flow.squeeze(0)
#     indices = np.array(flow.shape[1:]) // 2
#     i, j, k = indices

#     slice_x_1 = rotate(img1[i, :, :])
#     slice_y_1 = rotate(img1[:, j, :])
#     slice_z_1 = rotate(img1[:, :, k])
#     slice_x_2 = rotate(img2[i, :, :])
#     slice_y_2 = rotate(img2[:, j, :])
#     slice_z_2 = rotate(img2[:, :, k])
#     slice_x_flow = (flow[1:3, i, :, :])
#     slice_x_flow_col = rotate(flow_vis.flow_to_color(
#         slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
#     slice_y_flow_col = rotate(flow_vis.flow_to_color(
#         slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     slice_z_flow = (flow[0:2, :, :, k])
#     slice_z_flow_col = rotate(flow_vis.flow_to_color(
#         slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     kwargs = {}
#     kwargs['cmap'] = 'gray'
#     x_extent, y_extent, z_extent = [(0, b - 1) for b in flow.shape[1:]]
#     axes[0][0].imshow(slice_x_1, extent=y_extent + z_extent, **kwargs)
#     axes[0][1].imshow(slice_y_1, extent=x_extent + z_extent, **kwargs)
#     axes[0][2].imshow(slice_z_1, extent=x_extent + y_extent, **kwargs)
#     axes[1][0].imshow(slice_x_2, extent=y_extent + z_extent, **kwargs)
#     axes[1][1].imshow(slice_y_2, extent=x_extent + z_extent, **kwargs)
#     axes[1][2].imshow(slice_z_2, extent=x_extent + y_extent, **kwargs)
#     axes[2][0].imshow(slice_x_flow_col, extent=y_extent + z_extent)
#     axes[2][1].imshow(slice_y_flow_col, extent=x_extent + z_extent)
#     axes[2][2].imshow(slice_z_flow_col, extent=x_extent + y_extent)
#     plt.tight_layout()

#     if output_path is not None and fig is not None:
#         fig.savefig(output_path, format='jpg')
#     if show:
#         plt.show()
#     # return slice_x_flow_col, slice_y_flow_col, slice_z_flow_col
#     return fig

# def plot_validation_fig(img1, img2, flow_gt, flow,
#                         axes=None,
#                         output_path=None,
#                         show=True, ):
#     fig = None
#     if axes is None:
#         fig, axes = plt.subplots(4, 3, figsize=(10, 10))

#     while len(img1.shape) > 3:
#         img1 = img1.squeeze(0)
#     while len(img2.shape) > 3:
#         img2 = img2.squeeze(0)
#     while len(flow_gt.shape) > 4:
#         flow_gt = flow_gt.squeeze(0)
#     while len(flow.shape) > 4:
#         flow = flow.squeeze(0)
#     indices = np.array(flow.shape[1:]) // 2
#     i, j, k = indices

#     slice_x_1 = rotate(img1[i, :, :])
#     slice_y_1 = rotate(img1[:, j, :])
#     slice_z_1 = rotate(img1[:, :, k])
#     slice_x_2 = rotate(img2[i, :, :])
#     slice_y_2 = rotate(img2[:, j, :])
#     slice_z_2 = rotate(img2[:, :, k])
#     slice_x_flow = (flow[1:3, i, :, :])
#     slice_x_flow_col = rotate(flow_vis.flow_to_color(
#         slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
#     slice_y_flow_col = rotate(flow_vis.flow_to_color(
#         slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     slice_z_flow = (flow[0:2, :, :, k])
#     slice_z_flow_col = rotate(flow_vis.flow_to_color(
#         slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     slice_x_flow_gt = (flow_gt[1:3, i, :, :])
#     slice_x_flow_col_gt = rotate(flow_vis.flow_to_color(
#         slice_x_flow_gt.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     slice_y_flow_gt = (torch.stack((flow_gt[0, :, j, :], flow_gt[2, :, j, :])))
#     slice_y_flow_col_gt = rotate(flow_vis.flow_to_color(
#         slice_y_flow_gt.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
#     slice_z_flow_gt = (flow_gt[0:2, :, :, k])
#     slice_z_flow_col_gt = rotate(flow_vis.flow_to_color(
#         slice_z_flow_gt.permute([1, 2, 0]).numpy(), convert_to_bgr=False))

#     kwargs = {}
#     kwargs['cmap'] = 'gray'
#     x_extent, y_extent, z_extent = [(0, b - 1) for b in flow.shape[1:]]
#     axes[0][0].imshow(slice_x_1, extent=y_extent + z_extent, **kwargs)
#     axes[0][1].imshow(slice_y_1, extent=x_extent + z_extent, **kwargs)
#     axes[0][2].imshow(slice_z_1, extent=x_extent + y_extent, **kwargs)
#     axes[1][0].imshow(slice_x_2, extent=y_extent + z_extent, **kwargs)
#     axes[1][1].imshow(slice_y_2, extent=x_extent + z_extent, **kwargs)
#     axes[1][2].imshow(slice_z_2, extent=x_extent + y_extent, **kwargs)
#     axes[2][0].imshow(slice_x_flow_col, extent=y_extent + z_extent)
#     axes[2][1].imshow(slice_y_flow_col, extent=x_extent + z_extent)
#     axes[2][2].imshow(slice_z_flow_col, extent=x_extent + y_extent)
#     axes[3][0].imshow(slice_x_flow_col_gt, extent=y_extent + z_extent)
#     axes[3][1].imshow(slice_y_flow_col_gt, extent=x_extent + z_extent)
#     axes[3][2].imshow(slice_z_flow_col_gt, extent=x_extent + y_extent)
#     plt.tight_layout()

#     if output_path is not None and fig is not None:
#         fig.savefig(output_path)
#     if show:
#         plt.show()
#     # return slice_x_flow_col, slice_y_flow_col, slice_z_flow_col
#     return fig

# def plot_warped_img(img1, img1_recons, axes=None, output_path=None, show=False):
#     fig = None
#     if axes is None:
#         fig, axes = plt.subplots(1, 3, figsize=(10, 10))
#     while len(img1.shape) > 3:
#         img1 = img1.squeeze(0)
#     while len(img1_recons.shape) > 3:
#         img1_recons = img1_recons.squeeze(0)
#     indices = np.array(img1.shape) // 2
#     i, j, k = indices
#     slice_x_r = rotate(img1[i, :, :])
#     slice_x_g = rotate(img1_recons[i, :, :])
#     slice_x_b = (slice_x_r+slice_x_g)/2
#     slice_x = np.dstack((slice_x_r, slice_x_g, slice_x_b))

#     slice_y_r = rotate(img1[:, j, :])
#     slice_y_g = rotate(img1_recons[:, j, :])
#     slice_y_b = (slice_y_r+slice_y_g)/2
#     slice_y = np.dstack((slice_y_r, slice_y_g, slice_y_b))
    
#     slice_z_r = rotate(img1[:, :, k])
#     slice_z_g = rotate(img1_recons[:, :, k])
#     slice_z_b = (slice_z_r+slice_z_g)/2
#     slice_z = np.dstack((slice_z_r, slice_z_g, slice_z_b))

#     kwargs = {}
#     x_extent, y_extent, z_extent = [(0, b - 1) for b in img1.shape]
#     f0 = axes[0].imshow(slice_x, extent=y_extent + z_extent, **kwargs)
#     f1 = axes[1].imshow(slice_y, extent=x_extent + z_extent, **kwargs)
#     f2 = axes[2].imshow(slice_z, extent=x_extent + y_extent, **kwargs)

#     img = np.concatenate([slice_x, slice_y, slice_z],axis=1)
#     plt.tight_layout()
#     if output_path is not None and fig is not None:
#         fig.savefig(output_path)
#     if show:
#         plt.show()
#     return fig

# def rotate(image):
#     return np.rot90(image)

def add_mask(p_warped, template_seg_map, seg_reconst, split_index):
    while len(template_seg_map.shape) > 3:
        template_seg_map = template_seg_map.squeeze(0)
    while len(seg_reconst.shape) > 3:
        seg_reconst = seg_reconst.squeeze(0)
    
    indices = np.array(template_seg_map.shape) // split_index 
    i, j, k = indices.astype(np.int)
    slice_x_r = rotate(template_seg_map[i, :, :])
    slice_x_g = rotate(seg_reconst[i, :, :])

    slice_y_r = rotate(template_seg_map[:, j, :])
    slice_y_g = rotate(seg_reconst[:, j, :])
    
    slice_z_r = rotate(template_seg_map[:, :, k])
    slice_z_g = rotate(seg_reconst[:, :, k])

    slice_x_2 = rotate(template_seg_map[i, :, :])
    slice_y_2 = rotate(template_seg_map[:, j, :])
    slice_z_2 = rotate(template_seg_map[:, :, k])

    seg_row = np.concatenate([slice_x_2, slice_y_2, slice_z_2],axis=1)
    seg_recons_row = np.concatenate([slice_x_g, slice_y_g, slice_z_g],axis=1).astype(np.float32)

    seg_arr = np.concatenate([seg_recons_row, seg_row, seg_recons_row], axis=0) # row1 has the required image to be segmented, row2 is the GT, row3 is the row2 warped to match row1
    seg_arr_rgb = cv2.cvtColor(seg_arr.astype(bool).astype(np.float32), cv2.COLOR_GRAY2RGB)

    p_warped_cropped = p_warped[0,:seg_arr.shape[1],:,:]
    np.place(p_warped_cropped[:,:,0], seg_arr.astype(bool), seg_arr_rgb[:,:,0])

    return np.expand_dims(p_warped_cropped,0)

def disp_warped_img(img1, img1_recons, img2, split_index=2):
    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img1_recons.shape) > 3:
        img1_recons = img1_recons.squeeze(0)
    while len(img2.shape) > 3:
        img2 = img2.squeeze(0)
    
    indices = np.array(img1.shape) // split_index 
    i, j, k = indices.astype(np.int)
    slice_x_r = img1[i, :, :]
    slice_x_g = img1_recons[i, :, :]
    slice_x_b = (slice_x_r+slice_x_g)/2
    slice_x = np.dstack((slice_x_r, slice_x_g, slice_x_b))

    slice_y_r = img1[:, j, :]
    slice_y_g = img1_recons[:, j, :]
    slice_y_b = (slice_y_r+slice_y_g)/2
    slice_y = np.dstack((slice_y_r, slice_y_g, slice_y_b))
    
    slice_z_r = img1[:, :, k]
    slice_z_g = img1_recons[:, :, k]
    slice_z_b = (slice_z_r+slice_z_g)/2
    slice_z = np.dstack((slice_z_r, slice_z_g, slice_z_b))

    slice_x_2 = img2[i, :, :]
    slice_y_2 = img2[:, j, :]
    slice_z_2 = img2[:, :, k]
    img2_row = np.concatenate([slice_x_2, slice_y_2, slice_z_2],axis=1)
    img2_row = cv2.cvtColor(img2_row,cv2.COLOR_GRAY2RGB)

    pink_green_row = np.concatenate([slice_x, slice_y, slice_z],axis=1)[None,::].squeeze(0)
    img_row = np.concatenate([slice_x_r, slice_y_r, slice_z_r],axis=1)
    img_row = cv2.cvtColor(img_row,cv2.COLOR_GRAY2RGB)
    img_recons_row = np.concatenate([slice_x_g, slice_y_g, slice_z_g],axis=1).astype(np.float32)
    img_recons_row = cv2.cvtColor(img_recons_row,cv2.COLOR_GRAY2RGB)

    img_arr = np.concatenate([img_row, img2_row, img_recons_row, pink_green_row],axis=0)
    
    return np.expand_dims(img_arr,0)

def disp_training_fig(img1, img2, flow):
    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img2.shape) > 3:
        img2 = img2.squeeze(0)
    while len(flow.shape) > 4:
        flow = flow.squeeze(0)
    indices = np.array(flow.shape[1:]) // 2
    i, j, k = indices

    slice_x_1 = img1[i, :, :]
    slice_y_1 = img1[:, j, :]
    slice_z_1 = img1[:, :, k]
    slice_x_2 = img2[i, :, :]
    slice_y_2 = img2[:, j, :]
    slice_z_2 = img2[:, :, k]
    
    slice_x_flow = (flow[1:3, i, :, :])
    slice_x_flow_col = flow_vis.flow_to_color(slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False)
    slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
    slice_y_flow_col = flow_vis.flow_to_color(slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False)
    slice_z_flow = (flow[0:2, :, :, k])
    slice_z_flow_col = flow_vis.flow_to_color(slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False)
    slices_1 = [np.tile(sl_,(3, 1, 1)) for sl_ in [slice_x_1, slice_y_1, slice_z_1]]
    slices_2 = [np.tile(sl_,(3, 1, 1)) for sl_ in [slice_x_2, slice_y_2, slice_z_2]]
    flows12  = [np.transpose(fl_,(2, 0, 1)) for fl_ in [slice_x_flow_col, slice_y_flow_col, slice_z_flow_col]]

    slice_imgs = [np.concatenate([s1*255.,s2*255.,f12.astype(np.float32)], axis=1) for s1, s2, f12 in zip(slices_1,slices_2,flows12)]
    return np.floor(np.concatenate(slice_imgs,axis=2)[None,::])

def write_flow_as_nrrd(flow, folderpath='.', filename="flow.nrrd"): 
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    nrrd.write(os.path.join(folderpath,"x"+filename), flow[0,:,:,:])
    nrrd.write(os.path.join(folderpath,"y"+filename), flow[1,:,:,:])
    nrrd.write(os.path.join(folderpath,"z"+filename), flow[2,:,:,:])