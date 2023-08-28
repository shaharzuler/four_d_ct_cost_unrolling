import os

import numpy as np
import nrrd


def write_flow_as_nrrd(flow:np.ndarray, folderpath:str='.', filename:str="flow.nrrd") -> None: 
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    nrrd.write(os.path.join(folderpath,"x"+filename), flow[0,:,:,:])
    nrrd.write(os.path.join(folderpath,"y"+filename), flow[1,:,:,:])
    nrrd.write(os.path.join(folderpath,"z"+filename), flow[2,:,:,:])

