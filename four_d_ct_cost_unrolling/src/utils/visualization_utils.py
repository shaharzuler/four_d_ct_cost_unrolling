import numpy as np
import flow_vis
import numpy as np
import nrrd
import os
import cv2


def extract_img_middle_slices(img:np.array) -> tuple[np.array]:
    i, j, k = np.array(img.shape) // 2
    slice_x = img[i, :, :]
    slice_y = img[:, j, :]
    slice_z = img[:, :, k]
    return slice_x, slice_y, slice_z

def add_mask(img_warped:np.array, template_seg:np.array, seg_reconst:np.array) -> np.array:
    slice_x_2, slice_y_2, slice_z_2 = extract_img_middle_slices(template_seg)
    slice_x_g, slice_y_g, slice_z_g = extract_img_middle_slices(seg_reconst)

    seg_recons_row = np.concatenate([slice_x_g, slice_y_g, slice_z_g],axis=1).astype(np.float32)
    seg_template_row = np.concatenate([slice_x_2, slice_y_2, slice_z_2], axis=1)
    seg_recons_row_copy = seg_recons_row.copy()
    
    seg_arr = np.concatenate([seg_recons_row, seg_template_row, seg_recons_row_copy], axis=0) # row1 has the query unabeled image, row2 is the GT, row3 is the row2 warped to match row1
    seg_arr_rgb = cv2.cvtColor(seg_arr.astype(bool).astype(np.float32), cv2.COLOR_GRAY2RGB)

    img_warped_cropped = img_warped[:seg_arr.shape[0],:,:].astype(np.float32)
    np.copyto(img_warped_cropped[:,:,0], seg_arr_rgb[:,:,0],  where=seg_arr.astype(bool))
    return np.expand_dims(img_warped_cropped, 0)

def disp_warped_img(img1:np.array, img1_recons:np.array, img2:np.array) -> np.array:    
    slice_x_r, slice_y_r, slice_z_r = extract_img_middle_slices(img1)
    slice_x_g, slice_y_g, slice_z_g = extract_img_middle_slices(img1_recons)
    slice_x_b, slice_y_b, slice_z_b = (slice_x_r + slice_x_g)/2, (slice_y_r+slice_y_g)/2, (slice_z_r+slice_z_g)/2

    slice_x = np.dstack((slice_x_r, slice_x_g, slice_x_b))
    slice_y = np.dstack((slice_y_r, slice_y_g, slice_y_b))
    slice_z = np.dstack((slice_z_r, slice_z_g, slice_z_b))

    img_row = np.concatenate([slice_x_r, slice_y_r, slice_z_r], axis=1).astype(np.float32)
    img_row = cv2.cvtColor(img_row, cv2.COLOR_GRAY2RGB)
    img_row = cv2.putText(img_row, "template_image", org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(1.,0,0), thickness=2)

    slice_x_2, slice_y_2, slice_z_2 = extract_img_middle_slices(img2)
    img2_row = np.concatenate([slice_x_2, slice_y_2, slice_z_2], axis=1).astype(np.float32)
    img2_row = cv2.cvtColor(img2_row, cv2.COLOR_GRAY2RGB)
    img2_row = cv2.putText(img2_row, "unlabeled_image", org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(1.,0,0), thickness=2)

    img_recons_row = np.concatenate([slice_x_g, slice_y_g, slice_z_g],axis=1).astype(np.float32)
    img_recons_row = cv2.cvtColor(img_recons_row,cv2.COLOR_GRAY2RGB)
    img_recons_row = cv2.putText(img_recons_row, "warped_image", org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(1.,0,0), thickness=2)

    colored_error_row = np.concatenate([slice_x, slice_y, slice_z],axis=1)[None,::].squeeze(0)
    colored_error_row = cv2.putText(colored_error_row, "images_diff", org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(1.,0,0), thickness=2)

    img_arr = np.concatenate([img_row, img2_row, img_recons_row, colored_error_row], axis=0)
    
    return np.expand_dims(img_arr, 0)

def extract_flow_middle_slices(flow:np.array) -> list[tuple[np.array]]:
    return [extract_img_middle_slices(flow[i,:,:,:]) for i in range(3)]
    
def disp_training_fig(img1:np.array, img2:np.array, flow:np.array) -> np.array:
    slice_x_1, slice_y_1, slice_z_1 = extract_img_middle_slices(img1)
    slice_x_2, slice_y_2, slice_z_2 = extract_img_middle_slices(img2)

    slices_1 = [np.tile(slice,(3, 1, 1)) for slice in [slice_x_1, slice_y_1, slice_z_1]]
    slices_2 = [np.tile(slice,(3, 1, 1)) for slice in [slice_x_2, slice_y_2, slice_z_2]]

    flows12 = disp_flow_colors(flow)

    slice_imgs = [np.concatenate([slice_1, slice_2, slice_flow.astype(np.float32)/255], axis=1) for slice_1, slice_2, slice_flow in zip(slices_1, slices_2, flows12)]

    return np.concatenate(slice_imgs,axis=2)[None,::]

def get_2d_flow_sections(flow:np.array) -> tuple[np.array]:
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

def _get_most_contours_from_hirarchies(contours:tuple) -> np.array:
    most_contours = np.zeros([0])
    for contours_level in contours:
        if contours_level.shape[0] > most_contours.shape[0]:
            most_contours = contours_level
    return most_contours

def get_mask_contours(mask:np.array, downsample_factor:int=2) -> np.array:
    contours, hierarchy = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = _get_most_contours_from_hirarchies(contours)
    contours = contours[::downsample_factor,:,:]
    return contours

def _add_flow_contour_arrows(image:np.array, contours:np.array, slice_flow:np.array, arrow_scale_factor:int=5, equal_arrow_length:bool=False) -> np.array:
    for contour in contours:
        start, end = _get_arrow_start_end_coords(contour, slice_flow, arrow_scale_factor, equal_arrow_length)
        image = cv2.arrowedLine(image,(start[0],start[1]),(end[0],end[1]),color=(0,0,0),thickness=1)
    return image

def _get_arrow_start_end_coords(contour:np.array, slice_flow:np.array, arrow_scale_factor:int, equal_arrow_length:bool):
    start = contour[0]
    delta = slice_flow[:, contour[0,1], contour[0,0]]
    if equal_arrow_length:
        delta /= np.linalg.norm(delta, 2)
    end = np.round(start+delta*arrow_scale_factor).astype(start.dtype)
    return start, end

def _add_arrows_from_mask_on_2d_img(img_slice:np.array, mask_slice:np.array, flow_slice:np.array) -> np.array:
    contours = get_mask_contours(mask_slice)        
    img_slice_w_arrows = _add_flow_contour_arrows(img_slice, contours, flow_slice)
    return img_slice_w_arrows

def disp_flow_as_arrows(img:np.array, seg:np.array, flow:np.array, text:str=None) -> np.array:
    img_slices_gray = extract_img_middle_slices(img)
    img_slice_x, img_slice_y, img_slice_z = [cv2.cvtColor(slice.astype(np.float32),cv2.COLOR_GRAY2RGB) for slice in img_slices_gray]
    mask_x_1, mask_y_1, mask_z_1 = extract_img_middle_slices(seg)
    slice_x_flow, slice_y_flow, slice_z_flow = get_2d_flow_sections(flow)

    slice_x_w_arrows = _add_arrows_from_mask_on_2d_img(img_slice_x, mask_x_1, slice_x_flow)
    slice_y_w_arrows = _add_arrows_from_mask_on_2d_img(img_slice_y, mask_y_1, slice_y_flow)
    slice_z_w_arrows = _add_arrows_from_mask_on_2d_img(img_slice_z, mask_z_1, slice_z_flow)

    all_flow_arrowed_disp = np.concatenate([slice_x_w_arrows, slice_y_w_arrows, slice_z_w_arrows], axis=1)
    if text is not None:
        all_flow_arrowed_disp = cv2.putText(all_flow_arrowed_disp, text, org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(1.,0,0), thickness=2)
    all_flow_arrowed_disp = np.expand_dims(np.transpose(all_flow_arrowed_disp, (2,0,1)), 0)
    return all_flow_arrowed_disp

def disp_flow_colors(flow:np.array) -> np.array:
    slice_x_flow, slice_y_flow, slice_z_flow = get_2d_flow_sections(flow)

    slice_x_flow = np.transpose(slice_x_flow,[1, 2, 0])
    slice_y_flow = np.transpose(slice_y_flow,[1, 2, 0])
    slice_z_flow = np.transpose(slice_z_flow,[1, 2, 0])

    slice_x_flow_col = flow_vis.flow_to_color(slice_x_flow, convert_to_bgr=False)
    slice_y_flow_col = flow_vis.flow_to_color(slice_y_flow, convert_to_bgr=False)
    slice_z_flow_col = flow_vis.flow_to_color(slice_z_flow, convert_to_bgr=False)

    flows_colors  = [np.transpose(slice,(2, 0, 1)) for slice in [slice_x_flow_col, slice_y_flow_col, slice_z_flow_col]]
    return flows_colors

def _disp_single_flow_colors(flow:np.array, text:str=None) -> np.array:
    flow_disp = np.dstack(disp_flow_colors(flow)).astype(np.float32)/255
    if text is not None:
        flow_disp = np.transpose(flow_disp, (1,2,0))
        flow_disp = cv2.putText(flow_disp, text, org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(1.,0,0), thickness=2)
        flow_disp = np.transpose(flow_disp, (2,0,1))
    return flow_disp

def disp_flow_error_colors(flows_pred:np.array, flows_gt:np.array) -> np.array:
    flows_pred_disp = _disp_single_flow_colors(flows_pred, text="prediction")
    flows_gt_disp = _disp_single_flow_colors(flows_gt, text="ground truth")
    diff = flows_pred - flows_gt
    abs_diff_disp = _disp_single_flow_colors(diff, text="error")
    abs_diff = np.abs(diff)
    abs_flow_diff = np.sum(np.dstack(get_2d_flow_sections(abs_diff)), axis=0) #clip? TODO #TODO maybe add epe map here
    abs_flow_diff_rgb = cv2.cvtColor(abs_flow_diff.astype(np.float32), cv2.COLOR_GRAY2RGB)
    abs_flow_diff_rgb = cv2.putText(abs_flow_diff_rgb, "absolute error", org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(1.,0,0), thickness=2)
    abs_flow_diff_rgb = np.transpose(abs_flow_diff_rgb, (2,0,1))

    flow_error_colors_fig = np.concatenate((flows_pred_disp, flows_gt_disp, abs_diff_disp, abs_flow_diff_rgb ),axis=1)[None,::]
    return flow_error_colors_fig

