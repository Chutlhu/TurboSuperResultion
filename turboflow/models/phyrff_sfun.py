import torch
import torch.nn as nn
import torch.nn.functional as F

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
        num_layers = len(dim_layers)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearTanh(dim_layers[l], dim_layers[l+1]))
            
        blocks.append(nn.Linear(dim_layers[-2], dim_layers[-1]))
        # potential is positive! (is it?)
        blocks.append(nn.Tanh())
        self.network = nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.network(x)
        return x


class SfunRFFNet(nn.Module):
    
    def __init__(self, name, dim_mpl_layers, 
                 f_nfeatures, f_scale, 
                 smallest_increment,
                 n_centers = 20,
                 lam_pde=1):
        super(SfunRFFNet, self).__init__()
        self.name = name
        
        # regression/pinn network       
        self.rff = Fourier(f_nfeatures, f_scale) # directly the random matrix 'cause of checkpoint and load
        self.mlp = MLP(dim_mpl_layers)
        
        # offgrid patches params
        self.n_centers = n_centers
        self.increment_pixels = [1, 2, 3, 4] # in pixels wrt the hight resolution (=256)
        self.min_l = smallest_increment
        self.patch_dim = 64
        
        # pde params
        self.lam_pde = lam_pde
        
    
    def forward(self, xin): # x := BxC(Batch, InputChannels)
        ## Fourier features
        x = self.rff(xin) # Batch x Fourier Features
        ## MLP
        x = self.mlp(x)
        return x

    def offgrid_patches(self, x):
        
        # for earch 
        C = torch.randn(self.n_centers,2)

        # make a patch
        patch_ln = torch.arange(-self.min_l*self.patch_dim, self.min_l*self.patch_dim, self.min_l, device=x.device)
        # make it square
        patch_sq = torch.stack(torch.meshgrid(patch_ln, patch_ln), dim=-1)

        for c in range(self.n_centers):
            # center the patches
            pass



    def fit(self, trainloader, epochs=1000):
        self.train()
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)
        epoch = 0
        while epoch < epochs or loss < 1e-6:
            epoch += 1
            current_loss = 0
            batches = 0
            for x_batch, y_batch in trainloader:
                batches += 1
                optimiser.zero_grad()
                y_hat = self.forward(x_batch)

                # check div=0
                u, v = torch.split(y_hat,1,-1)
                du_xy = torch.autograd.grad(u, x_batch, torch.ones_like(u), create_graph=True)[0]       
                dv_xy = torch.autograd.grad(v, x_batch, torch.ones_like(v), create_graph=True)[0]
                div_u_xy = du_xy[...,0] + dv_xy[...,1]
                loss_pde = (1/batches)*torch.norm(div_u_xy)**2
            
                # rec loss
                loss_rec = (1/batches)*F.mse_loss(y_hat, y_batch)

                loss = loss_rec + self.lam_pde*loss_pde
                current_loss +=  loss.item() - current_loss


                loss.backward()
                optimiser.step()
                if epoch % 100 == 0:
                    print('Epoch: %d, Loss: (rec: [%f] + %1.2f * div-free: [%f]) = %f' %
                     (epoch, loss_rec.item(), self.lam_pde, loss_pde.item(), current_loss))

        print('Done with Training')
        print('Final error:', current_loss)