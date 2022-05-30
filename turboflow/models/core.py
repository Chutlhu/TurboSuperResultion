import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from orthnet import Legendre

pi = 3.14159265359

class CNN(nn.Module):
    """ A simple 5 layer CNN, configurable by passing a hyperparameter dictionary at initialization.
        Based upon the one outlined in the Pytorch intro tutorial 
        (http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network)
    """
  
    def __init__(self, RFFx, RFFt, hyperparam_dict=None):
        super(CNN, self).__init__()
        
        if not hyperparam_dict :
            hyperparam_dict = self.standard_hyperparams()
        
        self.hyperparam_dict = hyperparam_dict

        self.conv1 = nn.Sequential(
                nn.Conv2d(1, hyperparam_dict['conv1_size'], 3)
            ,   nn.GELU()
            ,   nn.MaxPool2d(2, 2)
        )
        self.conv2 =  nn.Sequential(
                nn.Conv2d(hyperparam_dict['conv1_size'], hyperparam_dict['conv2_size'], 3)
            ,   nn.GELU()
            ,   nn.MaxPool2d(2, 2)
        )

        # self.conv3 = nn.Sequential(
        #         nn.Conv2d(hyperparam_dict['conv2_size'], hyperparam_dict['conv3_size'], 3)
        #     ,   nn.GELU()
        #     ,   nn.MaxPool2d(2, 2)
        # )
        
        # self.conv4 = nn.Sequential(
        #         nn.Conv2d(hyperparam_dict['conv3_size'], hyperparam_dict['conv4_size'], 3)
        #     ,   nn.GELU()
        #     ,   nn.MaxPool2d(2, 2)
        # )


        self.fc1 = nn.Linear(
            hyperparam_dict['fc1_in_size'], 
            hyperparam_dict['fc1_out_size']
        )
        self.mlp = nn.Sequential(self.fc1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.mlp(x))
        return x

    def standard_hyperparams(self):
        hyperparam_dict = {}
        
        hyperparam_dict['conv1_size'] = 8
        hyperparam_dict['conv2_size'] = 16
        hyperparam_dict['conv3_size'] = 32
        hyperparam_dict['conv4_size'] = 64

        hyperparam_dict['fc1_in_size'] = 61504
        hyperparam_dict['fc1_out_size'] = 256
        
        return hyperparam_dict
        

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        # v = self.norm(v)
        v = self.spatial_proj(v).squeeze()
        out = u * v
        return out


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, 1)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        # x = self.norm(x) # norm axis = channel
        # x = F.tanh(self.channel_proj1(x))
        x = torch.tanh(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out

class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)


class GaborFilter(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta=1.0, max_freq=128.):
        super(GaborFilter, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim)

        # Init weights
        self.linear.weight.data *= max_freq * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        norm = (x ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(x))


class FourierFilter(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta=2.0, bias=True, max_freq=128.):
        super(FourierFilter, self).__init__()

        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # Init weights
        self.linear.weight.data *= max_freq * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return torch.sin(self.linear(x))


class FreqProj(nn.Module):

    def __init__(self, nrfft):
        super(FreqProj, self).__init__()

        freqs_1D = torch.linspace(0,1,nrfft).float()
        self.F = nrfft
        self.k = torch.stack(torch.meshgrid([freqs_1D, freqs_1D])).flatten(start_dim=1)
        # coeff = torch.randn(2,nrfft,nrfft,2)
        # self.coeff = nn.Parameter(coeff)

    def forward(self, f, x, t):
        '''
        f = B x Nout x F (= 2 x 2 x F^2)
        x = B x Nin
        '''
        B, D = x.shape

        k = self.k.to(x.device)
        norm = x @ k # X x K
        D = torch.exp(1j*2*pi*norm / self.F)
        # D = torch.linalg.pinv(D.T)  # X x K

        f = f.reshape(B, 2, self.F*self.F, 2) 
        f = torch.view_as_complex(f)
        # val = torch.unique(t)
        # for v in val:
        #     mask = t == v
        #     f[mask,:,:] = torch.mean(f[mask,:,:], dim=0) # B x 2 x KK
            
        u = torch.mean(f * D[:,None,:], dim=-1)
        return u.real
        

class OrthoPoly(nn.Module):
    def __init__(self, degree):
        super(OrthoPoly, self).__init__()

        self.degree = degree
        self.legendre = lambda x : Legendre(x, degree)
        self.linear = nn.Linear(1)

    def forward(self, x):
        '''
        x = B x Nin
        '''
        x = self.legendre(x).tensor # B x Nin -> B x D
        return x


class iFreqProj(nn.Module):
    def __init__(self, nrfft):
        super(iFreqProj, self).__init__()

        freqs_1D = torch.arange(nrfft)
        self.F = nrfft
        self.freqs = torch.stack(torch.meshgrid([freqs_1D, freqs_1D])).flatten(start_dim=1)

    def forward(self, f, x):
        '''
        f = B x Nout x F (= 2 x 2 x F^2)
        x = B x Nin
        '''
        B, D = x.shape
        x = x * self.F

        norm = x @ self.freqs.to(x.device)
        norm = norm.reshape(B, self.F, self.F)  # B x F x F

        f = f.reshape(B, 2, self.F, self.F, 2)
        f = torch.mean(f, dim=0)
        f = torch.view_as_complex(f)            # B x 2 x F x F
        
        u = torch.mean(f[None,...] * torch.exp(1j*2*pi*norm / self.F)[:,None,...], dim=[-1,-2]) # B x 2
        return u.real


class MFN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, k=4, filter_fun='Gabor', data_max_freq=128):
        super(MFN, self).__init__()

        if filter_fun == 'Gabor':
            filter_fun = GaborFilter
        else:
            filter_fun = FourierFilter

        self.k = k
        self.filters = nn.ModuleList(
            [filter_fun(in_dim, hidden_dim, alpha=6.0 / k, max_freq=data_max_freq) for _ in range(k)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(k - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])

        for lin in self.linear[:k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim))

    def forward(self, x):
        # Recursion - Equation 3
        zi = self.filters[0](x)  # Eq 3.a
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.filters[i + 1](x)  # Eq 3.b
        x = self.linear[self.k - 1](zi)  # Eq 3.c
        return x

class MLP(nn.Module):
    
    def __init__(self, dim_layers, last_activation_fun_name, do_last_activ=True):
        super(MLP, self).__init__()
        layers = []
        num_layers = len(dim_layers)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearTanh(dim_layers[l], dim_layers[l+1]))
            
        blocks.append(nn.Linear(dim_layers[-2], dim_layers[-1]))
        if do_last_activ:
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


class SineLayer(nn.Module):

    def __init__(self, w0):
        super(SineLayer, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self, w0=30, in_dim=2, hidden_dim=256, out_dim=1):
        super(Siren, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, out_dim))

        # Init weights
        with torch.no_grad():
            self.net[0].weight.uniform_(-1. / in_dim, 1. / in_dim)
            self.net[2].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[4].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[6].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[8].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[10].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)

    def forward(self, x):
        return self.net(x)



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
    
    def __init__(self, nfeat, scale, nvars=2):
        super().__init__()
        b = torch.randn(nvars, nfeat)*scale
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, x):
        x = torch.einsum('bc,cf->bf', 2*pi*x, self.b.to(x.device))
        return torch.cat([torch.sin(x), torch.cos(x)], -1)

class Fourier2(nn.Module):
    
    def __init__(self, nfeat, scale, nvars=2):
        super().__init__()
        self.b1 = nn.Parameter(torch.randn(nvars, nfeat)*scale, requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(nvars, nfeat)*1, requires_grad=False)

    def forward(self, x):
        x1 = torch.einsum('bc,cf->bf', 2*pi*x, self.b1.to(x.device))
        x2 = torch.einsum('bc,cf->bf', 2*pi*x, self.b2.to(x.device))
        return torch.cat([  torch.sin(x1) * torch.sin(x2), 
                            torch.cos(x1) * torch.cos(x2)], -1)

class TimeFourier(nn.Module):
    
    def __init__(self, nfeat_s, scale_s, nfeat_t, scale_t):
        super().__init__()
        self.b_s = nn.Parameter(torch.randn(2, nfeat_s)*scale_s, requires_grad=False)
        self.b_t = nn.Parameter(torch.randn(1, nfeat_t)*scale_t, requires_grad=False)

    def forward(self, x):
        t = x[:,:1]
        x = x[:,1:]
        x = torch.einsum('bc,cf->bf', 2*pi*x, self.b_s.to(x.device))
        t = torch.einsum('bc,cf->bf', 2*pi*t, self.b_t.to(t.device))
        return torch.cat([torch.sin(x), torch.cos(x), torch.sin(t), torch.cos(t)], -1)


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