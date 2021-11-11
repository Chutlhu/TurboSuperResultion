import numpy as np
import torch
from turboflow.utils import torch_utils as tch
from turboflow.utils import phy_utils as phy


def angular_error_2Dfield(x, x_ref, avg=True):
    assert x.shape == x_ref.shape
    assert x.shape[1] == 2
    N = x.shape[0]
    
    w1 = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
    w2 = torch.cat([x_ref, torch.ones(x_ref.shape[0], 1, device=x.device)], dim=1)
    
    err = torch.sum(w1 * w2, dim=1) / (torch.norm(w1, dim=1) * torch.norm(w2, dim=1))
    err[err > 1.] = 1.
    err[err <-1.] = -1.
    assert err.sum() <= N
    err = torch.rad2deg(torch.acos(err))
    if avg:
        return err.mean()
    return err

def recostruction_error(x, x_ref, avg=True):
    err = torch.norm(x - x_ref, dim=1)**2
    if avg:
        return torch.mean(err)
    return err


def spectum_logerror_2Dfield(x, x_ref, avg=True, eps=1e-30):
    assert x.shape == x_ref.shape
    assert x.shape[1] == 2
    
    R = int(x.shape[0]**.5)

    x_ = phy.energy_spectrum(x.view(R,R,2).permute(2,0,1))[0]
    x_ref_ = phy.energy_spectrum(x_ref.view(R,R,2).permute(2,0,1))[0]

    err = torch.abs(torch.log(x_+eps) - torch.log(x_ref_+eps))**2
    if avg:
        return torch.mean(err)
    return err

def compute_all_metrics(u_pred, u_ref, avg=True):

    err_rec = recostruction_error(u_pred, u_ref, avg)
    err_ang = angular_error_2Dfield(u_pred, u_ref, avg)
    err_spec = spectum_logerror_2Dfield(u_pred, u_ref, avg)

    metrics = {
        'reconstruction' : err_rec.item(),
        'angular_degree' : err_ang.item(),
        'log_err_specturm' : err_spec.item(),
    }

    return metrics



# def energy_spectrum(vel):
#     """
#     Compute energy spectrum given a velocity field
#     :param vel: tensor of shape (N, 3, res, res, res)
#     :return spec: tensor of shape(N, res/2)
#     :return k: tensor of shape (res/2,), frequencies corresponding to spec
#     """
#     device = vel.device
#     res = vel.shape[-2:]

#     assert(res[0] == res[1])
#     r = res[0]
#     k_end = int(r/2)
#     vel_ = pad_rfft3(vel, onesided=False) # (N, 3, res, res, res, 2)
#     uu_ = (torch.norm(vel_, dim=-1) / r**3)**2
#     e_ = torch.sum(uu_, dim=1)  # (N, res, res, res)
#     k = fftfreqs(res).to(device) # (3, res, res, res)
#     rad = torch.norm(k, dim=0) # (res, res, res)
#     k_bin = torch.arange(k_end, device=device).float()+1
#     bins = torch.zeros(k_end+1).to(device)
#     bins[1:-1] = (k_bin[1:]+k_bin[:-1])/2
#     bins[-1] = k_bin[-1]
#     bins = bins.unsqueeze(0)
#     bins[1:] += 1e-3
#     inds = torch.searchsorted(bins, rad.flatten().unsqueeze(0)).squeeze().int()
#     # bincount = torch.histc(inds.cpu(), bins=bins.shape[1]+1).to(device)
#     bincount = torch.bincount(inds)
#     asort = torch.argsort(inds.squeeze())
#     sorted_e_ = e_.view(e_.shape[0], -1)[:, asort]
#     csum_e_ = torch.cumsum(sorted_e_, dim=1)
#     binloc = torch.cumsum(bincount, dim=0).long()-1
#     spec_ = csum_e_[:,binloc[1:]] - csum_e_[:,binloc[:-1]]
#     spec_ = spec_[:, :-1]
#     spec_ = spec_ * 2 * np.pi * (k_bin.float()**2) / bincount[1:-1].float()
#     return spec_, k_bin

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