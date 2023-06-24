import numpy as np
import flow_vis
import numpy as np
import nrrd
import os
import cv2


def extract_img_middle_slices(img):
    i, j, k = np.array(img.shape) // 2
    slice_x = img[i, :, :]
    slice_y = img[:, j, :]
    slice_z = img[:, :, k]
    return slice_x, slice_y, slice_z


def add_mask(p_warped, template_seg_map, seg_reconst):
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