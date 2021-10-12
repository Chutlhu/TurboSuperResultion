import numpy as np
from turboflow.utils import torch_utils as tch

from turboflow.dataloaders import load_turbo2D_simple_numpy

def metrics(u_true, u_pred):

    pass

def results_potential_to_dict(Xlr, Xmr, Xhr, Ulr_gt, Umr_gt, Uhr_gt, model, device):
    ## predict velocity field
    xlr = tch.to_torch(Xlr, device)
    xmr = tch.to_torch(Xmr, device)
    xhr = tch.to_torch(Xhr, device)
    model.eval().to(device)
    ulr, Plr = model(xlr)
    umr, Pmr = model(xmr)
    uhr, Phr = model(xhr)

    ## back to numpy
    xlr = tch.to_numpy(xlr)
    ulr = tch.to_numpy(ulr)
    Plr = tch.to_numpy(Plr)

    xmr = tch.to_numpy(xmr)
    umr = tch.to_numpy(umr)
    Pmr = tch.to_numpy(Pmr)

    xhr = tch.to_numpy(xhr)
    uhr = tch.to_numpy(uhr)
    Phr = tch.to_numpy(Phr)

    L = int(Xlr.shape[0]**0.5)
    M = int(Xmr.shape[0]**0.5)
    H = int(Xhr.shape[0]**0.5)

    loss_rec = np.array(model.loss_rec)
    loss_pde = np.array(model.loss_pde)
    loss_tot = loss_rec + loss_pde

    res_dict = {
        'name' : model.name,
        'LR' : {
            'x' : xlr,
            'u' : ulr,
            'u_gt' : Ulr_gt,
            'P' : Plr,
            'size' : L
        },
        'MR' : {
            'x' : xmr,
            'u' : umr,
            'u_gt' : Umr_gt,
            'P' : Pmr,
            'size' : M
        },
        'HR' : {
            'x' : xhr,
            'u' : uhr,
            'u_gt' : Uhr_gt,
            'P' : Phr,
            'size' : H
        },
        'loss' : {
            'lam_pde' : model.lam_pde,
            'rec' : loss_rec,
            'pde' : loss_pde,
            'tot' : loss_tot,
        }
    }
    return res_dict