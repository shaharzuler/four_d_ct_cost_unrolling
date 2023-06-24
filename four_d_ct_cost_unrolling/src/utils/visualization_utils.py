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


def extract_img_middle_slices(img):
    i, j, k = np.array(img.shape) // 2
    slice_x = img[i, :, :]
    slice_y = img[:, j, :]
    slice_z = img[:, :, k]
    return slice_x, slice_y, slice_z


def add_mask(p_warped, template_seg_map, seg_reconst):
    # while len(template_seg_map.shape) > 3:
    #     template_seg_map = template_seg_map.squeeze(0)
    # while len(seg_reconst.shape) > 3:
    #     seg_reconst = seg_reconst.squeeze(0)
    slice_x_2, slice_y_2, slice_z_2 = extract_img_middle_slices(template_seg_map)
    slice_x_g, slice_y_g, slice_z_g = extract_img_middle_slices(seg_reconst)

    seg_template_row = np.concatenate([slice_x_2, slice_y_2, slice_z_2], axis=1)
    seg_recons_row = np.concatenate([slice_x_g, slice_y_g, slice_z_g],axis=1).astype(np.float32)

    seg_arr = np.concatenate([seg_recons_row, seg_template_row, seg_recons_row], axis=0) # row1 has the query unabeled image, row2 is the GT, row3 is the row2 warped to match row1
    seg_arr_rgb = cv2.cvtColor(seg_arr.astype(bool).astype(np.float32), cv2.COLOR_GRAY2RGB)

    p_warped_cropped = p_warped[:seg_arr.shape[0],:,:]
    np.place(p_warped_cropped[:,:,0], seg_arr.astype(bool), seg_arr_rgb[:,:,0])

    return np.expand_dims(p_warped_cropped, 0)

def disp_warped_img(img1, img1_recons, img2):
    # while len(img1.shape) > 3:
    #     img1 = img1.squeeze(0)
    # while len(img1_recons.shape) > 3:
    #     img1_recons = img1_recons.squeeze(0)
    # while len(img2.shape) > 3:
    #     img2 = img2.squeeze(0)
    
    slice_x_r, slice_y_r, slice_z_r = extract_img_middle_slices(img1)
    slice_x_g, slice_y_g, slice_z_g = extract_img_middle_slices(img1_recons)
    slice_x_b, slice_y_b, slice_z_b = (slice_x_r + slice_x_g)/2, (slice_y_r+slice_y_g)/2, (slice_z_r+slice_z_g)/2

    slice_x = np.dstack((slice_x_r, slice_x_g, slice_x_b))
    slice_y = np.dstack((slice_y_r, slice_y_g, slice_y_b))
    slice_z = np.dstack((slice_z_r, slice_z_g, slice_z_b))

    img_row = np.concatenate([slice_x_r, slice_y_r, slice_z_r], axis=1).astype(np.float32)
    img_row = cv2.cvtColor(img_row, cv2.COLOR_GRAY2RGB)

    slice_x_2, slice_y_2, slice_z_2 = extract_img_middle_slices(img2)
    img2_row = np.concatenate([slice_x_2, slice_y_2, slice_z_2], axis=1).astype(np.float32)
    img2_row = cv2.cvtColor(img2_row, cv2.COLOR_GRAY2RGB)

    img_recons_row = np.concatenate([slice_x_g, slice_y_g, slice_z_g],axis=1).astype(np.float32)
    img_recons_row = cv2.cvtColor(img_recons_row,cv2.COLOR_GRAY2RGB)

    colored_error_row = np.concatenate([slice_x, slice_y, slice_z],axis=1)[None,::].squeeze(0)

    img_arr = np.concatenate([img_row, img2_row, img_recons_row, colored_error_row], axis=0)
    
    return np.expand_dims(img_arr,0)



def extract_flow_middle_slices(flow):
    return [extract_img_middle_slices(flow[i,:,:,:]) for i in range(3)]
    
def disp_training_fig(img1, img2, flow):
    # while len(img1.shape) > 3:
    #     img1 = img1.squeeze(0)
    # while len(img2.shape) > 3:
    #     img2 = img2.squeeze(0)
    # while len(flow.shape) > 4:
    #     flow = flow.squeeze(0)
    slice_x_1, slice_y_1, slice_z_1 = extract_img_middle_slices(img1)
    slice_x_2, slice_y_2, slice_z_2 = extract_img_middle_slices(img2)

    slice_x_flow, slice_y_flow, slice_z_flow = get_2d_flow_sections(flow)

    slice_x_flow = np.transpose(slice_x_flow,[1, 2, 0])
    slice_y_flow = np.transpose(slice_y_flow,[1, 2, 0])
    slice_z_flow = np.transpose(slice_z_flow,[1, 2, 0])

    slice_x_flow_col = flow_vis.flow_to_color(slice_x_flow, convert_to_bgr=False)
    slice_y_flow_col = flow_vis.flow_to_color(slice_y_flow, convert_to_bgr=False)
    slice_z_flow_col = flow_vis.flow_to_color(slice_z_flow, convert_to_bgr=False)

    slices_1 = [np.tile(slice,(3, 1, 1)) for slice in [slice_x_1, slice_y_1, slice_z_1]]
    slices_2 = [np.tile(slice,(3, 1, 1)) for slice in [slice_x_2, slice_y_2, slice_z_2]]
    flows12  = [np.transpose(slice,(2, 0, 1)) for slice in [slice_x_flow_col, slice_y_flow_col, slice_z_flow_col]]

    slice_imgs = [np.concatenate([slice_1*255.,slice_2*255.,slice_flow.astype(np.float32)], axis=1) for slice_1, slice_2, slice_flow in zip(slices_1, slices_2, flows12)]
    return np.round(np.concatenate(slice_imgs,axis=2)[None,::])

def get_2d_flow_sections(flow:np.array) -> np.array:
    slices_of_x_flow, slices_of_y_flow, slices_of_z_flow = extract_flow_middle_slices(flow)
    slice_x_flow = np.stack((slices_of_y_flow[0], slices_of_z_flow[0]))
    slice_y_flow = np.stack((slices_of_x_flow[1], slices_of_z_flow[1]))
    slice_z_flow = np.stack((slices_of_x_flow[2], slices_of_y_flow[2]))
    return slice_x_flow, slice_y_flow, slice_z_flow

def write_flow_as_nrrd(flow, folderpath='.', filename="flow.nrrd"): 
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    nrrd.write(os.path.join(folderpath,"x"+filename), flow[0,:,:,:])
    nrrd.write(os.path.join(folderpath,"y"+filename), flow[1,:,:,:])
    nrrd.write(os.path.join(folderpath,"z"+filename), flow[2,:,:,:])

def _get_most_contours_from_hirarchies(contours):
    most_contours = np.zeros([0])
    for contours_level in contours:
        if contours_level.shape[0] > most_contours.shape[0]:
            most_contours = contours_level
    return most_contours

def get_mask_contours(mask:np.array, downsample_factor:int=2):
    contours, hierarchy = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = _get_most_contours_from_hirarchies(contours)
    contours = contours[::downsample_factor,:,:]
    return contours

def _add_flow_contour_arrows(image:np.array, contours:np.array, slice_flow:np.array, arrow_scale_factor:int=5, equal_arrow_length:bool=False):
    for contour in contours:
        start, end = _get_arrow_start_end_coords(contour, slice_flow, arrow_scale_factor, equal_arrow_length)
        image = cv2.arrowedLine(image,(start[0],start[1]),(end[0],end[1]),color=(0,0,0),thickness=1)
    return image

def _get_arrow_start_end_coords(contour, slice_flow, arrow_scale_factor, equal_arrow_length):
    start = contour[0]
    delta = slice_flow[:, contour[0,1], contour[0,0]]
    if equal_arrow_length:
        delta /= np.linalg.norm(delta, 2)
    end = np.round(start+delta*arrow_scale_factor).astype(start.dtype)
    return start, end

def _add_arrows_from_mask_on_2d_img(img_slice, mask_slice, flow_slice):
    contours = get_mask_contours(mask_slice)        
    img_slice_w_arrows = _add_flow_contour_arrows(img_slice, contours, flow_slice)
    return img_slice_w_arrows

def disp_flow_as_arrows(img:np.array, seg:np.array, flow:np.array) -> np.array:
    img_slices_gray = extract_img_middle_slices(img)
    img_slice_x, img_slice_y, img_slice_z = [cv2.cvtColor(slice.astype(np.float32),cv2.COLOR_GRAY2RGB) for slice in img_slices_gray]
    mask_x_1, mask_y_1, mask_z_1 = extract_img_middle_slices(seg)
    slice_x_flow, slice_y_flow, slice_z_flow = get_2d_flow_sections(flow)

    slice_x_w_arrows = _add_arrows_from_mask_on_2d_img(img_slice_x, mask_x_1, slice_x_flow)
    slice_y_w_arrows = _add_arrows_from_mask_on_2d_img(img_slice_y, mask_y_1, slice_y_flow)
    slice_z_w_arrows = _add_arrows_from_mask_on_2d_img(img_slice_z, mask_z_1, slice_z_flow)

    all_flow_arrowed_disp = np.concatenate([slice_x_w_arrows, slice_y_w_arrows, slice_z_w_arrows], axis=1)
    all_flow_arrowed_disp = np.expand_dims(np.transpose(all_flow_arrowed_disp, (2,0,1)), 0)
    return all_flow_arrowed_disp