import torch
import numpy as np

def torch_to_np(tensor:torch.tensor) -> np.array:
    return tensor.detach().cpu().numpy()