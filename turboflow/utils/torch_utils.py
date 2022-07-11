from itertools import accumulate
import torch

def get_device(verbose=True):
    torch.cuda.is_available()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if verbose: print('Torch running on:', device)
    return device


def to_numpy(x):
    return x.detach().cpu().numpy()


def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)


def diff(f, x, order=1):
    assert x.requires_grad
    if order == 1:
        return torch.autograd.grad(f, x, torch.ones_like(f), retain_graph=True, create_graph=True,)[0]
    if order == 2:
        return torch.autograd.grad(diff(f, x), x, torch.ones_like(f), retain_graph=True, create_graph=True, )[0]


def _my_field_grad(f, dim):
    """
    dim = 1 : derivative wrt x direct
    dim = 2 : derivative wrt y direct
    courtesy from https://github.com/Rose-STL-Lab/Turbulent-Flow-Net/
    """
    assert f.shape[0] == f.shape[1] # input must be in shape (R,R)
    R = f.shape[0]
    dx = 1/R
    dim += 1
    D = 2
    assert D == len(f.shape)
    out = torch.zeros_like(f)
    
    # initialize slices
    slice1 = [slice(None)]*D
    slice2 = [slice(None)]*D
    slice3 = [slice(None)]*D
    slice4 = [slice(None)]*D

    # 2nd order interior
    slice1[-dim] = slice(1, -1)
    slice2[-dim] = slice(None, -2)
    slice3[-dim] = slice(1, -1)
    slice4[-dim] = slice(2, None)
    out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2*dx)
    
    # 2nd order edges
    slice1[-dim] = 0
    slice2[-dim] = 0
    slice3[-dim] = 1
    slice4[-dim] = 2
    a = -1.5 / dx
    b = 2. / dx
    c = -0.5 / dx
    out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    slice1[-dim] = -1
    slice2[-dim] = -3
    slice3[-dim] = -2
    slice4[-dim] = -1
    a = 0.5 / dx
    b = -2. / dx
    c = 1.5/ dx

    out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    return out

def set_grad(nn_module, grad_val):
    for param in nn_module.parameters():
        param.requires_grad = grad_val

def to_np(x) : return x.detach().cpu().numpy()
