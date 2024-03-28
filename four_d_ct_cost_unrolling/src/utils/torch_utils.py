from typing import Dict, OrderedDict, Tuple
import torch
import shutil
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np



def weight_parameters(module):
    return [param for name, param in module.named_parameters() if 'weight' in name]

def bias_parameters(module):
    return [param for name, param in module.named_parameters() if 'bias' in name]

def load_checkpoint(model_path:str) -> Tuple[int,OrderedDict]:
    weights = torch.load(model_path, map_location={f'cuda:{i}': 'cpu' for i in range(8)})
    epoch = None
    if 'epoch' in weights:
        epoch = weights.pop('epoch')
    if 'state_dict' in weights:
        state_dict = (weights['state_dict'])
    else:
        state_dict = weights
    return epoch, state_dict

def save_checkpoint(save_dir:str, states:Dict, prefix:str, is_best:bool, filename:str='ckpt.pth.tar') -> None:
    file_path = os.path.join(save_dir, f'{prefix}_{filename}')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(states, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(save_dir, f'{prefix}_model_best.pth.tar'))

def rescale_mask_tensor(mask:torch.Tensor, scale_factor:Tuple) -> torch.Tensor:
    if scale_factor != (1, 1, 1):
        mask_scaled = F.interpolate(mask.float(), scale_factor=scale_factor, mode='nearest').bool()
    else:
        mask_scaled = mask
    return mask_scaled

def torch_to_np(tensor:torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def mask_xyz_to_13xyz(mask:torch.tensor) -> torch.tensor: 
    return mask.repeat(1,3,1,1,1)

def torch_nd_dot(A, B, axis):
    mult = A*B
    return torch.sum(mult,axis=axis)
