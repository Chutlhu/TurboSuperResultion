import torch
import torch.nn as nn
import torch.nn.functional as F

pi = 3.14159265359

class MLP(nn.Module):
    
    def __init__(self, dim_layers, last_activation_fun_name):
        super(MLP, self).__init__()
        layers = []
        num_layers = len(dim_layers)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearTanh(dim_layers[l], dim_layers[l+1]))
            
        blocks.append(nn.Linear(dim_layers[-2], dim_layers[-1]))
        blocks.append(self.last_activation_function(last_activation_fun_name))
        self.network = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.network(x)

    @staticmethod
    def last_activation_function(fun_name:str)->nn.Module:
        if fun_name == 'tanh':
            return nn.Tanh()
        if fun_name == 'sigmoid':
            return nn.Sigmoid()
        if fun_name == 'relu':
            return nn.ReLU()
        if fun_name == 'id':
            return nn.Identity()
        if fun_name == 'ELU':
            return nn.ELU()
        return None


class DivFree2DBasis(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        # self.k = torch.abs(nn.Parameter(torch.randint(-scale, scale, size=(1,nfeat)), requires_grad=False))
        # self.l = torch.abs(nn.Parameter(torch.randint(-scale, scale, size=(1,nfeat)), requires_grad=False))
        self.k = nn.Parameter(torch.arange(1,nfeat+1), requires_grad=False)[None,:]
        self.l = nn.Parameter(torch.arange(1,nfeat+1), requires_grad=False)[None,:]
        self.nfeat = nfeat

    def forward(self, x):

        k = self.k[:,:,None].to(x.device)
        l = self.l[:,None,:].to(x.device)

        y = x[:,1][:,None,None]
        x = x[:,0][:,None,None]
        
        a = l*(x**k)*(y**(l-1))  # B x K x L
        b = -k*(x**(k-1))*(y**l) # B x K x L

        # mask_k = torch.nonzero(k==0)
        # a[mask_k] = y[mask_k]**l
        # b[mask_k] = 0

        # mask_l = torch.nonzero(l==0)
        # a[mask_l] = 0
        # b[mask_l] = x[mask_l]**k

        a = a.view(x.shape[0],self.nfeat**2)
        b = b.view(x.shape[0],self.nfeat**2)
        
        return torch.cat([a, b], -1)


class Fourier(nn.Module):
    
    def __init__(self, nfeat, scale):
        super().__init__()
        self.b = nn.Parameter(torch.randn(2, nfeat)*scale, requires_grad=False)

    def forward(self, x):
        x = torch.einsum('bc,cf->bf', 2*pi*x, self.b.to(x.device))
        return torch.cat([torch.sin(x), torch.cos(x)], -1)


class MTL(nn.Module):

    def __init__(self):
        super(MTL, self).__init__()

        self.log_sigma_sqr_rec = nn.Parameter(torch.Tensor([2])).float()
        self.log_sigma_sqr_sdiv = nn.Parameter(torch.Tensor([2])).float()
        self.log_sigma_sqr_sfn = nn.Parameter(torch.Tensor([2])).float()
        self.log_sigma_sqr_grads = nn.Parameter(torch.Tensor([2])).float()

    def forward(self, x):
        return x

def LinearReLU(n_in, n_out):
    # do not work with ModuleList here either.
    block = nn.Sequential(
      nn.Linear(n_in, n_out),
      nn.ReLU()
    )
    return block


def LinearTanh(n_in, n_out):
    # do not work with ModuleList here either.
    block = nn.Sequential(
      nn.Linear(n_in, n_out),
      nn.Tanh()
    )
    return block


def make_offgrid_patches_xcenter(n_centers:int, min_l:float, patch_dim:float, device):
    """
    for each random point in the image, make a square patch
    return: C x P x P x 2
    """
    
    # for earch 
    if n_centers is None:
        n_centers = n_centers
    a = patch_dim*min_l
    b = 1 - patch_dim*min_l
    centers = ((b - a)*torch.rand(n_centers,2) + a).to(device)

    ## make a patch
    # define one axis
    patch_ln = torch.arange(-min_l*patch_dim, min_l*patch_dim, min_l, device=device)
    # make it square meshgrid
    patch_sq = torch.stack(torch.meshgrid(patch_ln, patch_ln), dim=-1)
    
    ## center the patch for all the centers
    size = (n_centers, *patch_sq.shape)
    patch_sq_xcenter = patch_sq.unsqueeze(0).expand(size)
    assert torch.allclose(patch_sq_xcenter[0,:,:], patch_sq)
    assert torch.allclose(patch_sq_xcenter[-1,:,:], patch_sq)
    patch_sq_xcenter = patch_sq_xcenter + centers[:,None,None,:]
    # some checks
    assert len(patch_sq_xcenter.shape) == 4
    assert patch_sq_xcenter.shape[-1] == 2
    assert patch_sq_xcenter.shape[0] == n_centers
    assert patch_sq_xcenter.shape[1] == patch_sq_xcenter.shape[2] == patch_dim*2
    return patch_sq_xcenter


def make_offgrid_patches_xcenter_xincrement(n_increments:int, n_centers:int, min_l:float, patch_dim:float, device):
    """
    for each random point in the image and for each increments, make a square patch
    return: I x C x P x P x 2
    """
    patches_xcenter = make_offgrid_patches_xcenter(n_centers, min_l, patch_dim, device) # C x P x P x 2
    increments = min_l * torch.arange(0,n_increments,device=patches_xcenter.device)   
    # expand patches for each increments
    size = (n_increments, *patches_xcenter.shape)
    patches_xcenter_xincrement = patches_xcenter.unsqueeze(0).expand(size)
    assert torch.allclose(patches_xcenter_xincrement[0,:,:], patches_xcenter)
    assert torch.allclose(patches_xcenter_xincrement[1,:,:], patches_xcenter)
    patches_xcenter_xincrement = patches_xcenter_xincrement + increments[:,None,None,None,None]
    # some checks
    assert len(patches_xcenter_xincrement.shape) == 5
    assert patches_xcenter_xincrement.shape[-1] == 2
    assert patches_xcenter_xincrement.shape[0] == n_increments
    assert patches_xcenter_xincrement.shape[1] == n_centers
    assert patches_xcenter_xincrement.shape[2] == patches_xcenter_xincrement.shape[3] == patch_dim*2
    return patches_xcenter_xincrement


def montecarlo_sampling_xcenters_xincerments(n_points, n_increments, n_neighbours, min_l, device):
    
    random_points = torch.rand(n_points, 2, device=device).to(float)
    random_direction = 2*torch.rand(n_neighbours, n_points, device=device).to(float) - 1
    
    traslated_points = torch.zeros(n_increments, n_neighbours, n_points, 2, device=device)
    for l in range(n_increments):
        traslated_points[l,:,:,0] = min_l*l*torch.cos(2*pi*random_direction) + random_points[None,:,0]
        traslated_points[l,:,:,1] = min_l*l*torch.sin(2*pi*random_direction) + random_points[None,:,1]

    return traslated_points