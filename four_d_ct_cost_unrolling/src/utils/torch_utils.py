import torch
import shutil
import torch.nn.functional as F


def weight_parameters(module):
    return [param for name, param in module.named_parameters() if 'weight' in name]

def bias_parameters(module):
    return [param for name, param in module.named_parameters() if 'bias' in name]

def load_checkpoint(model_path):
    weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    epoch = None
    if 'epoch' in weights:
        epoch = weights.pop('epoch')
    if 'state_dict' in weights:
        state_dict = (weights['state_dict'])
    else:
        state_dict = weights
    return epoch, state_dict

def save_checkpoint(save_dir, states, prefix, is_best, filename='ckpt.pth.tar'):
    file_path = save_dir / f'{prefix}_{filename}'
    torch.save(states, file_path)
    if is_best:
        shutil.copyfile(file_path, save_dir / f'{prefix}_model_best.pth.tar')


def rescale_mask_tensor(mask:torch.tensor, scale_factor:tuple):
    if scale_factor != (1,1,1):
        mask_scaled = F.interpolate(mask.float(), scale_factor=scale_factor, mode='nearest').bool()
    else:
        mask_scaled = mask
    return mask_scaled