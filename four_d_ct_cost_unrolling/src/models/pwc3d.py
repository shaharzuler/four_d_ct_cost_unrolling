import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pwc_blocks import FeatureExtractor, Correlation, FlowEstimatorReduce, ContextNetwork, conv
from .admm.admm import ADMMSolverBlock, MaskGenerator
from ..utils.flow_utils import rescale_flow_tensor, flow_warp

class PWC3D(nn.Module):
    def __init__(self, args:dict, upsample:bool=True, search_range:int=4): 
        super(PWC3D, self).__init__()
        self.search_range = search_range
        self.num_chs = [1, 16, 32, 64, 96, 128, 192]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.n_frames = 2
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)

        self.upsample = upsample

        self.corr = Correlation(pad_size=self.search_range, kernel_size=1,
                                max_displacement=self.search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.dim_corr = (self.search_range * 2 + 1) ** 3 
        self.num_ch_in = 32 + (self.dim_corr + 2) * (self.n_frames - 1) + 1

        self.flow_estimators = FlowEstimatorReduce(self.num_ch_in)

        self.context_networks = ContextNetwork( (self.flow_estimators.feat_dim + 2) * (self.n_frames - 1) + 1 )

        self.admm_block = ADMMSolverBlock(rho=args.admm_args.rho, lamb=args.admm_args.lamb, eta=args.admm_args.eta, grad=args.admm_args.grad, T=args.admm_args.T)
        self.mask_gen = MaskGenerator(alpha=args.admm_args.alpha, learn_mask=args.admm_args.learn_mask)
        self.apply_admm = args.admm_args.apply_admm

        self.conv_1x1 = nn.ModuleList([conv(192, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96,  32, kernel_size=1, stride=1, dilation=1),
                                       conv(64,  32, kernel_size=1, stride=1, dilation=1),
                                       conv(32,  32, kernel_size=1, stride=1, dilation=1)])


    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self, layer):
        if isinstance(layer, nn.Conv3d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, data:dict[str,torch.tensor]) -> dict[str,torch.tensor]:
        x1, x2, vox_dim = data["img1"], data["img2"], data["vox_dim"]
        self._calculate_pyramid_reduction(x1)
        x1_p = self.feature_pyramid_extractor(x1) + [x1] 
        x2_p = self.feature_pyramid_extractor(x2) + [x2]
        out_scale = 2**(self.num_levels - self.output_level - 1) 
        masks = [self.mask_gen(x_, vox_dim/out_scale, scale=1/out_scale) for x_ in [x1, x2]]

        res_dict = {
            'flows_fw': self._forward_2_frames(x1_p, x2_p, mask=masks[0], vox_dim=vox_dim)
        }
        return res_dict

    def _calculate_pyramid_reduction(self, x1):
        self.num_levels_to_reduce_from_pyramid = 0
        self.feature_pyramid_extractor.num_levels_to_reduce_from_pyramid = self.num_levels_to_reduce_from_pyramid
        max_allowed_pyramid_levels = int(math.log2(min(x1.shape[2:]))) 
        if max_allowed_pyramid_levels < self.num_levels:
            self.num_levels_to_reduce_from_pyramid = self.num_levels - max_allowed_pyramid_levels
            self.feature_pyramid_extractor.num_levels_to_reduce_from_pyramid = self.num_levels_to_reduce_from_pyramid

    def _forward_2_frames(self, x1_p: torch.Tensor, x2_p: torch.Tensor, mask, vox_dim):
        flows = []
        aux_vars = {
                "q":        [],
                "c":        [],
                "betas":    [],
                "masks":    mask
            }
        
        N, C, H, W, D = x1_p[0].size()
        init_dtype = x1_p[0].dtype 
        init_device = x1_p[0].device
        flow12 = torch.zeros(N, 3, H, W, D, dtype=init_dtype, device=init_device).float()


        for l, (_x1, _x2) in enumerate(zip(x1_p, x2_p)):
            # warping
            if l == 0:
                x2_warp = _x2
            else:
                flow12 = rescale_flow_tensor(flow12, _x2.shape)
                x2_warp = flow_warp(_x2, flow12)

            # correlation
            out_corr = self.corr(_x1, x2_warp)
            out_corr_relu = self.leakyRELU(out_corr) 

            # concat and estimate flow
            x1_1by1 = self.conv_1x1[l+self.num_levels_to_reduce_from_pyramid](_x1) 

            x_intm, flow_res = self.flow_estimators(torch.cat([out_corr_relu, x1_1by1, flow12], dim=1)) 
            
            flow12 = flow12 + flow_res
            flow_fine = self.context_networks(torch.cat([x_intm, flow12], dim=1))
            flow12 = flow12 + flow_fine
            flows.append(flow12)
            
            if self.apply_admm[l+self.num_levels_to_reduce_from_pyramid]: 
                vox_dim_sc = vox_dim / 2**(self.num_levels - l + self.num_levels_to_reduce_from_pyramid - 1) 
                Q, C, Betas = self.admm_block(flow12, aux_vars["masks"], vox_dim_sc)
                aux_vars["q"].append(Q)
                aux_vars["c"].append(C)
                aux_vars["betas"].append(Betas)

            if l == self.output_level-self.num_levels_to_reduce_from_pyramid:
                break

        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4, mode='trilinear') for flow in flows]

        return flows[::-1], aux_vars

