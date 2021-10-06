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

    
class MLP(nn.Module):
    
    def __init__(self, dim_layers):
        super(MLP, self).__init__()
        layers = []
        num_layers = len(dim_layers)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearReLU(dim_layers[l], dim_layers[l+1]))
            
        blocks.append(nn.Linear(dim_layers[-2], dim_layers[-1]))
        blocks.append(nn.Tanh())
        self.network = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.network(x)


class PiRFFNet(nn.Module):
    
    def __init__(self, name, dim_mpl_layers, f_nfeatures, f_scale, delta_x):
        super(PiRFFNet, self).__init__()
        self.name = name
        
        self.lam_pde = 1e-4
        self.delta_x = delta_x

        # regression/pinn network       
        self.rff = Fourier(f_nfeatures, f_scale) # directly the random matrix 'cause of checkpoint and load
        self.mlp = MLP(dim_mpl_layers)
    
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
        while epoch < epochs or loss < 1e-6:
            epoch += 1
            current_loss = 0
            batches = 0
            progress = 0
            for x_batch, y_batch in trainloader:
                batches += 1
                optimiser.zero_grad()
                
                x_batch.requires_grad_(True)

                y_hat = self.forward(x_batch)

                # compute soft constraint
                u, v = torch.split(y_hat,1,-1)
                du_x = torch.autograd.grad(u, x_batch, torch.ones_like(u), create_graph=True)[0]       
                dv_y = torch.autograd.grad(v, x_batch, torch.ones_like(v), create_graph=True)[0]
                div_u = du_x[...,0] + dv_y[...,1]
                loss_pde = torch.norm(div_u)

                # compute divernge numerically
                xp1_y = x_batch + torch.Tensor([1,0])[None,:]
                xm1_y = x_batch - torch.Tensor([1,0])[None,:]
                x_yp1 = x_batch + torch.Tensor([0,1])[None,:]
                x_ym1 = x_batch - torch.Tensor([0,1])[None,:]
                u_xp1_y = self.forward(xp1_y)
                u_xm1_y = self.forward(xm1_y)
                u_x_yp1 = self.forward(x_yp1)
                u_x_ym1 = self.forward(x_ym1)
                hx = self.delta_x
                hy = self.delta_x
                div_u = (u_xp1_y - u_xm1_y)/(2*hx) + (u_x_yp1 - u_x_ym1)/(2*hy)
            
                loss_rec = F.mse_loss(y_hat, y_batch)
                loss = loss_rec + self.lam_pde*loss_pde

                current_loss += (1/batches) * (loss.item() - current_loss)


                loss.backward()
                optimiser.step()
                progress += y_batch.size(0)
                if epoch % 100 == 0:
                    print('Epoch: %d, Loss: (%f + %f) = %f' % (epoch, loss_rec.item(), loss_pde.item(), current_loss))
        print('Done with Training')
        print('Final error:', current_loss)