import argparse

import os
import numpy as np

# DNN Libraries
from torch.utils.data import DataLoader

# Turboflow Libraries
from turboflow.dataloaders import Turb2DDataset
from turboflow.models.phyrff_hard import DivFreeRFFNet
from turboflow.utils.torch_utils import get_device

def get_path_and_prepare_folder():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    return root_dir

# def get_args():

#     # these are project-wide arguments
#     parser = argparse.ArgumentParser(prog='main.py', add_help=True)
#     # add data specific arguments
#     args = parser.parse_args()
#     arg_groups={}

#     for group in parser._action_groups:
#         group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
#         arg_groups[group.title]=argparse.Namespace(**group_dict)
    
#     return arg_groups


if __name__ == '__main__':
    
    # args = get_args()
    
    # PREPARE FOLDER
    root_dir = get_path_and_prepare_folder()

    # DATA
    data_dir = '/home/dicarlo_d/Documents/Code/TurboSuperResultion/.cache/Turb2D.hdf5'
    trainset = Turb2DDataset(data_dir, ds=4, time_idx=42)
    trainloader = DataLoader(trainset, batch_size=100000, num_workers=16)

    # MODEL
    model = DivFreeRFFNet('DivFreeRFFNet', 
        dim_mpl_layers=[2]+3*[256]+[1],
        last_activation_fun_name='tanh',
        do_rff=True,f_nfeatures=256,f_scale=10,
        smallest_increment=0.00784314,n_increments=3,n_centers=30,
        lam_reg=0, lam_sfn=0, lam_pde=1, 
        verbose=True
    )

    # TRAIN
    device = get_device()
    model.to(device)
    model.fit(trainloader, epochs=1000)