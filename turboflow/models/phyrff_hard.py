import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from turboflow.models.basics import LinearReLU, LinearTanh, Fourier


class MLP(nn.Module):
    
    def __init__(self, dim_layers, last_activation_fun):
        super(MLP, self).__init__()
        layers = []
        num_layers = len(dim_layers)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearTanh(dim_layers[l], dim_layers[l+1]))
            
        blocks.append(nn.Linear(dim_layers[-2], dim_layers[-1]))
        blocks.append(last_activation_fun)
        self.network = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.network(x)

    
class MLPhard(nn.Module):
    
    def __init__(self, dim_layers, nfeat, scale):
        super(MLPhard, self).__init__()# intermidiate
        num_layers = len(dim_layers)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearTanh(dim_layers[l], dim_layers[l+1]))
            
        blocks.append(
            nn.Sequential(
                nn.Linear(dim_layers[-2], dim_layers[-1]),
                nn.Sigmoid() # potential is positive! (is it?)
            )
        )
        
        self.rff = Fourier(nfeat, scale)
        self.mlp = nn.Sequential(*blocks)
    
        self.sigma1 = lambda x : torch.tanh(x)
        self.dsigma1 = lambda x : 1 - torch.tanh(x)**2
        self.sigma2 = lambda x : torch.sigmoid(x)
        self.dsigma2 = lambda x : torch.sigmoid(x)*(1 - torch.sigmoid(x))
        
        self.sincos = lambda x : torch.cat([torch.sin(x), torch.cos(x)], -1)
        self.dsincos = lambda x : torch.cat([torch.cos(x), -torch.sin(x)], -1)

    def forward(self, x):
        x = self.rff(x)
        x = self.mlp(x)
        return x

    def get_wb(self, depth):
        return self.mlp[depth][0].weight, self.mlp[depth][0].bias
        
    def compute_ux(self, x):
        # Derivivate of the RFF
        Wr = self.rff.b
        Wr2 = torch.cat([Wr, Wr], axis=1).T
        
        dr = 2*self.rff.pi*self.dsincos(2*self.rff.pi*x @ Wr) # B x nfeat
        ar = self.sincos(2*self.rff.pi*x @ Wr) # B x nfeat
        # z = dr @ Wr

        # Derivate of the MLP
        W1, b1 = self.get_wb(0)
        d1 = self.dsigma1(ar @ W1.T + b1)
        a1 = self.sigma1(ar @ W1.T + b1) 
        # z = d1 @ W1
        z = ((d1 @ W1) * dr) @ Wr2
        
        W2, b2 = self.get_wb(1)
        d2 = self.dsigma1(a1 @ W2.T + b2)
        a2 = self.sigma1(a1 @ W2.T + b2)
        # z = (d2 @ W2) * d1) @ W1
        z = ((((d2 @ W2) * d1) @ W1) * dr) @ Wr2
        
        W3, b3 = self.get_wb(2)
        d3 = self.dsigma1(a2 @ W3.T + b3)
        a3 = self.sigma1(a2 @ W3.T + b3)
        # z = (d3 @ W3 * d2) @ W2 * d1) @ W1
        z = (((((d3 @ W3 * d2) @ W2) * d1) @ W1) * dr) @ Wr2
        
        W4, b4 = self.get_wb(3)
        d4 = self.dsigma2(a3 @ W4.T + b4)
        a4 = self.sigma2(a3 @ W4.T + b4)        
        # z = ((((d4 @ W4 * d3) @ W3 * d2) @ W2 * d1) @ W1
        z = ((((((d4 @ W4 * d3) @ W3 * d2) @ W2) * d1) @ W1) * dr) @ Wr2
        
        return z
    

class DivFree(nn.Module):

    def __init__(self, model):
        super(DivFree, self).__init__()


    def forward(self, f, xy):
        
        # its hardcoded for 2 variables both for input and fx
        assert f.shape[1] == 1
        assert xy.shape[1] == 2
        
        f_xy = self.compute_u_x(f, xy)
        div_free_uv = torch.cat([f_xy[:,1,None], 
                                -f_xy[:,0,None]], dim=-1)
        assert div_free_uv.shape[1] == 2
        assert div_free_uv.shape[0] == xy.shape[0]
        return div_free_uv


    def compute_u_x(self, f, x):
        x.requires_grad_(True)
        assert f.shape[1] == 1 # must be a scalar function
        f_x = torch.autograd.grad(f, x, torch.ones_like(f),
                                  create_graph=True,
                                  retain_graph=True)[0]
        assert f_x.shape == x.shape
        return f_x
    

class DivFreeRFFNet(nn.Module):
    
    def __init__(self, name, 
                    dim_mpl_layers, last_activation_fun,
                    do_rff, f_nfeatures, f_scale, 
                    smallest_increment, n_increments, n_centers,
                    lam_reg=1, lam_sfn=1, lam_pde=1,
                    verbose=True):

        super(DivFreeRFFNet, self).__init__()
        self.name = name
        self.verbose = verbose

        assert dim_mpl_layers[-1] == 1
        
        # regression/pinn network 
        self.do_rff = do_rff
        if do_rff:
            self.rff = Fourier(f_nfeatures, f_scale) # directly the random matrix 'cause of checkpoint and load
            dim_mpl_layers[0] = dim_mpl_layers[0]*f_nfeatures
        else:
            self.rff = None

        self.rff = Fourier(f_nfeatures, f_scale)
        self.mlp = MLP(dim_mpl_layers, last_activation_fun)

        # off-grid regularization params
        self.min_l = smallest_increment
        self.n_centers = n_centers
        self.patch_dim = 32
        self.n_increments = n_increments
        self.sfun = lambda x, y : torch.mean(torch.abs(x - y).pow(2))
        # self.sfun_model = 0.000275*smallest_increment*torch.arange(self.n_increments)**2
        self.sfun_model = 0.25*0.00275*torch.arange(self.n_increments)**2
        
        self.lam_pde = lam_pde
        self.lam_reg = lam_reg
        self.lam_sfn = lam_sfn

        self.loss = []
        self.loss_pde = []
        self.loss_rec = []
        self.loss_reg = []
        self.loss_sfn = []
        
    def forward(self, xin): # x := BxC(Batch, InputChannels)
        ## implement periodicity
        xin.requires_grad_(True)
        ## Fourier features
        if self.do_rff:
            x = self.rff(xin) # Batch x Fourier Features
            ## MLP
            x = self.mlp(x)
        else:
            x = self.mlp(xin)

        du_dxy = torch.autograd.grad(x, xin, torch.ones_like(x), create_graph=True)[0]

        div_free_uv = torch.cat([du_dxy[:,1,None], 
                                -du_dxy[:,0,None]], dim=-1)
        potential = x
        return div_free_uv, potential

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
                
                ## FORWARD
                y_hat, potential_hat = self.forward(x_batch)

                ## REGULARIZATION and LOSSES
                # L2 regularization on Potential
                loss_reg = (1/batches)*torch.norm(potential_hat)**2

                # Sfun regularization
                # 2.c Sfun-based loss
                x = self.make_offgrid_patches_xcenter_xincrement(x_batch.device)
                I, C, P, P, D = x.shape
                y_patches_hat = self.forward(x.reshape(I*C*P*P,D))[0].reshape(I, C, P, P, D)
                # compute structure function
                Sfun2 = torch.mean((y_patches_hat - y_patches_hat[0,...])**2, dim=[1,2,3,4])
                p = 1
                loss_sfn = torch.sum(torch.abs(torch.log(Sfun2+1e-10) - torch.log(self.sfun_model.to(x_batch.device)+1e-10))**p)

                # check div=0
                u, v = torch.split(y_hat,1,-1)
                du_xy = torch.autograd.grad(u, x_batch, torch.ones_like(u), create_graph=True)[0]       
                dv_xy = torch.autograd.grad(v, x_batch, torch.ones_like(v), create_graph=True)[0]
                div_u_xy = du_xy[...,0] + dv_xy[...,1]
                loss_pde = (1/batches)*torch.norm(div_u_xy)**2
            
                loss_rec = (1/batches)*F.mse_loss(y_hat, y_batch)

                loss = loss_rec + self.lam_pde*loss_pde + self.lam_reg*loss_reg + self.lam_sfn*loss_sfn
                current_loss +=  loss.item() - current_loss

                self.loss_rec.append(loss_rec.item())
                self.loss_pde.append(self.lam_pde*loss_pde.item())
                self.loss_reg.append(self.lam_reg*loss_reg.item())
                self.loss_sfn.append(self.lam_sfn*loss_sfn.item())

                ## BACKWARD
                loss.backward()
                optimiser.step()

                # LOG and PRINTS
                if self.verbose and (epoch % 100 == 0 or epoch == 1):
                    print('Epoch: %4d, Loss: (rec: [%1.4f] + df: [%1.4f] + regL2: [%1.4f] + regSfun: [%1.4f]) = %f' %
                     (epoch, loss_rec.item(), loss_pde.item(), loss_reg.item(), loss_sfn.item(), current_loss))

        print('Done with Training')
        print('Final error:', current_loss)


    def make_offgrid_patches_xcenter(self, device, n_centers = None):
        """
        for each random point in the image, make a square patch
        return: C x P x P x 2
        """
        
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

    def make_offgrid_patches_xcenter_xincrement(self, device):
        patches_xcenter = self.make_offgrid_patches_xcenter(device) # C x P x P x 2
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