# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from ..utils.flow_utils import flow_warp
# from .flow_loss import smooth_grad_1st



# class L2Loss(nn.modules.Module):
#     def __init__(self, args, ncc_win=5, w_ncc_scales=[1.0, 1.0, 1.0, 1.0, 1.0]):
#         super(L2Loss, self).__init__()
#         self.args = args
#         self.ncc_win = ncc_win
#         self.w_ncc_scales = w_ncc_scales
#         self.loss_operator = loss_operators[args.loss_operator_name]

#     def loss_smooth(self, flow, img1_scaled, vox_dim):

#         func_smooth = smooth_grad_1st


#         loss = []
#         loss += [func_smooth(flow, img1_scaled, vox_dim,
#                              self.args.alpha, flow_only=self.args.smooth_flow_only)]
#         return sum([l.mean() for l in loss])

#     def loss_admm(self, Q, C, Betas, w_admm):
#         loss = []

#         if w_admm > 0:
#             T = len(Q)
#             loss += [(q - c + beta)**2 / T for q,c,beta in zip(Q,C,Betas)]

#         return w_admm * self.args.admm_rho / 2 * sum([l.mean() for l in loss])

#     def forward(self, output, img1, img2, aux, vox_dim):
#         # log("Computing loss")
#         vox_dim = vox_dim.squeeze(0)

#         pyramid_flows = output
#         loss_ncc_func = self.loss_operator(win=self.ncc_win)
        
#         pyramid_smooth_losses = []
#         pyramid_ncc_losses = []
#         pyramid_admm_losses = []

#         aux12 = aux[0]
#         aux21 = aux[1]

#         pyramid_visible_mask1 = []
#         pyramid_visible_mask2 = []
#         pyramid_loss_ncc_viz = []

#         s = 1.
#         for i, flow in enumerate(pyramid_flows):
#             # log(f'Aggregating loss of pyramid level {i + 1}')
#             # log(f'Aggregating loss of pyramid level {i + 1}')

#             N, C, H, W, D = flow.size()

#             img1_scaled = F.interpolate(img1, (H, W, D), mode='area')
#             # Only needed if we aggregate flow21 and doing backward computation
#             img2_scaled = F.interpolate(img2, (H, W, D), mode='area')

#             flow12 = flow[:, :3]
#             flow21 = flow[:, 3:]
#             img1_recons = flow_warp(img2_scaled, flow12)
#             img2_recons = flow_warp(img1_scaled, flow21)

#             if self.args.w_occ:
#                 if i == 0:
#                     occu_mask1, occluded_ratio1 = get_occu_mask_bidirection(flow12, flow21, scale=1.85e+0) #############
#                     occu_mask2, occluded_ratio2 = get_occu_mask_bidirection(flow21, flow12, scale=1.85e+0) #############
#                     visible_mask1 = 1 - occu_mask1
#                     visible_mask2 = 1 - occu_mask2
                    
#                 else:
#                     visible_mask1 = F.interpolate(pyramid_visible_mask1[0],
#                                                (H, W, D), mode='nearest')
#                     visible_mask2 = F.interpolate(pyramid_visible_mask2[0],
#                                                (H, W, D), mode='nearest')
#             else:
#                 visible_mask1 = visible_mask2 = 1
#             pyramid_visible_mask1.append(visible_mask1)
#             pyramid_visible_mask2.append(visible_mask2)

#             if i == 0:
#                 s = min(H, W, D)

#             mask = (visible_mask1*visible_mask1).detach() ##################      
#             erosion_mask = torch.tensor(morphology.binary_erosion(mask.cpu().numpy(), selem=np.ones([1,1,3,3,3])),device=img1_scaled.device) ##################    
#             loss_ncc, loss_ncc_viz = loss_ncc_func(img1_scaled*visible_mask1, img1_recons*visible_mask1, erosion_mask)   #######

#             pyramid_loss_ncc_viz.append(loss_ncc_viz)
#             if self.args.w_sm_scales[i] >  0:
#                 loss_smooth = self.loss_smooth(flow=flow12 / s, img1_scaled=img1_recons, vox_dim=vox_dim)
#             else:
#                 loss_smooth = torch.zeros(1, device=loss_ncc.device)

#             if self.args.w_admm[i] > 0:
#                 loss_admm = self.loss_admm(aux12["q"][i], aux12["c"][i], aux12["betas"][i], self.args.w_admm[i])
#             else:
#                 loss_admm = torch.zeros(1, device=loss_ncc.device)

#             if self.args.w_bk:
#                 loss_ncc += loss_ncc_func(img2_scaled*visible_mask2, img2_recons*visible_mask2, erosion_mask)[0] ########
                
#                 if self.args.w_sm_scales[i] >  0:
#                     loss_smooth += self.loss_smooth(flow=flow21 / s, img1_scaled=img2_recons, vox_dim=vox_dim)

#                 if self.args.w_admm[i] > 0:
#                     loss_admm += self.loss_admm(aux21["q"][i], aux21["c"][i], aux21["betas"][i], self.args.w_admm[i])

#                 loss_smooth /= 2.
#                 loss_ncc /= 2.
#                 loss_admm /= 2.

#             log(f'Computed losses for level {i + 1}: loss_smoth={loss_smooth}'
#                 f'loss_ncc={loss_ncc}')

#             pyramid_smooth_losses.append(loss_smooth)
#             pyramid_ncc_losses.append(loss_ncc)
#             pyramid_admm_losses.append(loss_admm)

#         pyramid_smooth_losses = [l * w for l, w in
#                                  zip(pyramid_smooth_losses, self.args.w_sm_scales)]
#         pyramid_ncc_losses = [l * w for l, w in
#                               zip(pyramid_ncc_losses, self.w_ncc_scales)]
#         pyramid_admm_losses = [l * w for l, w in
#                               zip(pyramid_admm_losses, self.args.w_admm)]
#         log(f'Weighting losses')

#         loss_smooth = sum(pyramid_smooth_losses)
#         loss_ncc = sum(pyramid_ncc_losses)
#         loss_admm = sum(pyramid_admm_losses)
#         loss_total = loss_smooth + loss_ncc + loss_admm
#         pyramid_visible_masks = [(vis1, vis2) for vis1, vis2 in zip(pyramid_visible_mask1, pyramid_visible_mask2)]

#         return loss_total, loss_ncc, loss_smooth, loss_admm, pyramid_flows[0].abs().mean(), pyramid_visible_masks, pyramid_loss_ncc_viz

