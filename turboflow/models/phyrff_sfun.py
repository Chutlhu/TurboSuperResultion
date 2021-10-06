import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class Fourier(nn.Module):
    
    def __init__(self, nfeat, scale):
        super(Fourier, self).__init__()
        self.b = nn.Parameter(torch.randn(2, nfeat)*scale, requires_grad=False)
        self.pi = 3.14159265359

    def forward(self, x):
        x = torch.einsum('bc,cf->bf', 2*self.pi*x, self.b.to(x.device))
        return torch.cat([torch.sin(x), torch.cos(x)], -1)

    
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

    
class MLP(nn.Module):
    
    def __init__(self, dim_layers):
        super(MLP, self).__init__()
        layers = []
        num_layers = len(dim_layers)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearTanh(dim_layers[l], dim_layers[l+1]))
            
        blocks.append(nn.Linear(dim_layers[-2], dim_layers[-1]))
        blocks.append(nn.Tanh())
        self.network = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.network(x)


class SfunRFFNet(nn.Module):
    
    def __init__(self, name, dim_mpl_layers,
                f_nfeatures, f_scale,
                lam_pde, lam_sfun,
                smallest_increment, n_centers, n_increments):
        super(SfunRFFNet, self).__init__()
        self.name = name

        # pinn params
        self.lam_pde = lam_pde # 1e-4

        # regression/pinn network       
        self.rff = Fourier(f_nfeatures, f_scale) # directly the random matrix 'cause of checkpoint and load
        self.mlp = MLP(dim_mpl_layers)

        # off-grid regularization params
        self.min_l = smallest_increment
        self.n_centers = n_centers
        self.patch_dim = 32
        self.n_increments = n_increments
        self.sfun = lambda x, y : torch.mean(torch.abs(x - y).pow(2))
        # self.sfun_model = 0.000275*smallest_increment*torch.arange(self.n_increments)**2
        self.sfun_model = 0.25*0.00275*torch.arange(self.n_increments)**2
        self.lam_sfun = lam_sfun # 1e-3

    
    def forward(self, x): # x := BxC(Batch, InputChannels)
        ## implement periodicity
        x = torch.remainder(x,1)
        ## Fourier features
        x = self.rff(x) # Batch x Fourier Features
        ## MLP
        x = self.mlp(x)
        return x

    def fit(self, trainloader, epochs=1000):
        self.train()
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)
        epoch = 0

        self.sfun_model = self.sfun_model.to(self.get_device())
        C = self.n_centers
        I = self.n_increments


        while epoch < epochs or loss < 1e-6:
            epoch += 1
            current_loss = 0
            batches = 0
            progress = 0

            # TRAINING ON LR IMAGEs
            for x_batch, y_batch in trainloader:
                batches += 1
                
                x_batch.requires_grad_(True)

                # 1. FORWARD
                y_hat = self.forward(x_batch)

                # 2. LOSSes COMPUTATION
                # 2.a reconstruction loss 
                loss_rec = F.mse_loss(y_hat, y_batch)
                # 2.b soft constraint loss
                u, v = torch.split(y_hat,1,-1)
                du_x = torch.autograd.grad(u, x_batch, torch.ones_like(u), create_graph=True)[0]       
                dv_y = torch.autograd.grad(v, x_batch, torch.ones_like(v), create_graph=True)[0]
                div_u = du_x[...,0] + dv_y[...,1]
                loss_pde = torch.norm(div_u)
                # 2.c Sfun-based loss
                x = self.make_offgrid_patches_xcenter_xincrement()
                I, C, P, P, D = x.shape
                y_hat = self.forward(x.reshape(I*C*P*P,D)).reshape(I, C, P, P, D)
                # compute structure function
                Sfun2 = torch.mean((y_hat - y_hat[0,...])**2, dim=[1,2,3,4])
                p = 1
                loss_sfun = torch.sum(torch.abs(torch.log(Sfun2+1e-10) - torch.log(self.sfun_model+1e-10))**p)
                # 2.d total loss
                loss = loss_rec + self.lam_sfun*loss_sfun + self.lam_pde*loss_pde
                current_loss += (1/batches) * (loss.item() - current_loss)

                # BACKWARD
                optimiser.zero_grad()
                loss.backward()

                # GRADIEN DESC or ADAM STEP
                optimiser.step()

                # LOG PROGESS
                progress += y_batch.size(0)
                if epoch % 100 == 0:
                    print('Epoch: %d, Loss: (%f + %f + %f) = %f' % (epoch, loss_rec.item(), 
                                                        self.lam_pde*loss_pde.item(),
                                                        self.lam_sfun*loss_sfun.item(),
                                                        current_loss))

        print('Done with Training')
        print('Final error:', current_loss)


    def forward_patches(self,device):
        x = self.make_offgrid_patches(device)
        assert x.shape[-1] == 2
        size = x.shape
        x = x.view(-1,2)
        x = self.forward(x)
        x = x.view(*size)
        return x

    def get_device(self):
        return next(self.parameters()).device


    def make_offgrid_patches_xcenter(self, n_centers = None):
        """
        for each random point in the image, make a square patch
        return: C x P x P x 2
        """
        device = self.get_device()
        
        # for earch 
        if n_centers is None:
            n_centers = self.n_centers
        centers = torch.randn(n_centers,2).to(device)

        ## make a patch
        # define one axis
        patch_ln = torch.arange(-self.min_l*self.patch_dim, self.min_l*self.patch_dim, self.min_l, device=device)
        # make it square meshgrid
        patch_sq = torch.stack(torch.meshgrid(patch_ln, patch_ln), dim=-1)
        
        ## center the patch for all the centers
        size = (n_centers, *patch_sq.shape)
        patch_sq_xcenter = patch_sq.unsqueeze(0).expand(size)
        assert torch.allclose(patch_sq_xcenter[0,:,:], patch_sq)
        assert torch.allclose(patch_sq_xcenter[3,:,:], patch_sq)
        patch_sq_xcenter = patch_sq_xcenter + centers[:,None,None,:]
        # some checks
        assert len(patch_sq_xcenter.shape) == 4
        assert patch_sq_xcenter.shape[-1] == 2
        assert patch_sq_xcenter.shape[0] == n_centers
        assert patch_sq_xcenter.shape[1] == patch_sq_xcenter.shape[2] == self.patch_dim*2
        return patch_sq_xcenter


    def make_offgrid_patches_xcenter_xincrement(self):
        patches_xcenter = self.make_offgrid_patches_xcenter() # C x P x P x 2
        increments = self.min_l * torch.arange(0,self.n_increments,device=patches_xcenter.device)
        
        # expand patches for each increments
        size = (self.n_increments, *patches_xcenter.shape)
        patches_xcenter_xincrement = patches_xcenter.unsqueeze(0).expand(size)
        assert torch.allclose(patches_xcenter_xincrement[0,:,:], patches_xcenter)
        assert torch.allclose(patches_xcenter_xincrement[1,:,:], patches_xcenter)
        patches_xcenter_xincrement = patches_xcenter_xincrement + increments[:,None,None,None,None]
        # some checks
        assert len(patches_xcenter_xincrement.shape) == 5
        assert patches_xcenter_xincrement.shape[-1] == 2
        assert patches_xcenter_xincrement.shape[0] == self.n_increments
        assert patches_xcenter_xincrement.shape[1] == self.n_centers
        assert patches_xcenter_xincrement.shape[2] == patches_xcenter_xincrement.shape[3] == self.patch_dim*2
        return patches_xcenter_xincrement