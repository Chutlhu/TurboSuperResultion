import torch
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
    
    def __init__(self, name, dim_mpl_layers,
                    last_activation_fun,
                    do_rff, f_nfeatures, f_scale, lam_pde=1, 
                    hardcoded_divfree=False, verbose=True):
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

        self.hardcoded_divfree = hardcoded_divfree
        if self.hardcoded_divfree:
            self.mlp = MLPhard(dim_mpl_layers, f_nfeatures, f_scale)
        else:
            self.rff = Fourier(f_nfeatures, f_scale)
            self.mlp = MLP(dim_mpl_layers, last_activation_fun)
        
        self.lam_pde = lam_pde

        self.loss = []
        self.loss_pde = []
        self.loss_rec = []
        
    def forward(self, xin): # x := BxC(Batch, InputChannels)
        ## implement periodicity
        xin.requires_grad_(True)
        if self.hardcoded_divfree:
            ## RFF + MLP
            x = self.mlp(xin)
            ## DivFREE
            du_dxy = self.mlp.compute_ux(xin)
        else:
            ## Fourier features
            if self.do_rff:
                x = self.rff(xin) # Batch x Fourier Features
                ## MLP
                x = self.mlp(x)
                du_dxy = torch.autograd.grad(x, xin, torch.ones_like(x), create_graph=True)[0]
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
                y_hat, potential_hat = self.forward(x_batch)

                # check div=0
                u, v = torch.split(y_hat,1,-1)
                du_xy = torch.autograd.grad(u, x_batch, torch.ones_like(u), create_graph=True)[0]       
                dv_xy = torch.autograd.grad(v, x_batch, torch.ones_like(v), create_graph=True)[0]
                div_u_xy = du_xy[...,0] + dv_xy[...,1]
                loss_pde = (1/batches)*torch.norm(div_u_xy)**2
            
                loss_rec = (1/batches)*F.mse_loss(y_hat, y_batch)

                loss = loss_rec + self.lam_pde*loss_pde
                current_loss +=  loss.item() - current_loss

                self.loss_rec.append(loss_rec.item())
                self.loss_pde.append(self.lam_pde*loss_pde.item())

                loss.backward()
                optimiser.step()
                if self.verbose and (epoch % 100 == 0 or epoch == 1):
                    print('Epoch: %4d, Loss: (rec: [%f] + %1.2f * div-free: [%f]) = %f' %
                     (epoch, loss_rec.item(), self.lam_pde, loss_pde.item(), current_loss))

        print('Done with Training')
        print('Final error:', current_loss)