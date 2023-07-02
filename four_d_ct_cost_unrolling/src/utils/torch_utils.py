from typing import OrderedDict
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

def load_checkpoint(model_path:str) -> tuple[int,OrderedDict]:
    weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    epoch = None
    if 'epoch' in weights:
        epoch = weights.pop('epoch')
    if 'state_dict' in weights:
        state_dict = (weights['state_dict'])
    else:
        state_dict = weights
    return epoch, state_dict

def save_checkpoint(save_dir:str, states:dict, prefix:str, is_best:bool, filename:str='ckpt.pth.tar') -> None:
    file_path = os.path.join(save_dir, f'{prefix}_{filename}')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(states, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(save_dir, f'{prefix}_model_best.pth.tar'))

def rescale_mask_tensor(mask:torch.tensor, scale_factor:tuple) -> torch.tensor:
    if scale_factor != (1, 1, 1):
        mask_scaled = F.interpolate(mask.float(), scale_factor=scale_factor, mode='nearest').bool()
    else:
        mask_scaled = mask
    return mask_scaled

def torch_to_np(tensor:torch.tensor) -> np.array:
    return tensor.detach().cpu().numpy()