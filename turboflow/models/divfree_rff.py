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
        self.network = nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.network(x)
        return x

class DivFree(nn.Module):

    def __init__(self):
        super(DivFree, self).__init__()


    def forward(self, f, xy):
        
        # its hardcoded for 2 variables both for input and fx
        assert f.shape[1] == 1
        assert xy.shape[1] == 2
        
        # # compute first order deriv
        # # wrt xy
        # df_xy = torch.autograd.grad(f, xy, torch.ones_like(f), 
        #                 create_graph=True, 
        #                 retain_graph=True,
        #                 only_inputs=True)[0]
        # df_x, df_y = df_xy.split(1,-1)
        
        # # compute secord order deriv
        # df_x_xy = torch.autograd.grad(df_x, xy, torch.ones_like(df_x), 
        #                 create_graph=True, 
        #                 retain_graph=True,
        #                 only_inputs=True)[0]
        # df_xx, df_xy = df_x_xy.split(1,-1)
        # df_y_xy = torch.autograd.grad(df_y, xy, torch.ones_like(df_y),
        #                 create_graph=True, 
        #                 retain_graph=True,
        #                 only_inputs=True)[0]
        # df_yx, df_yy = df_y_xy.split(1,-1)

        # # # gather results in a matrix Bx2x2 in the form of Div-free kernel
        # # # by column
        # # K1 = torch.cat([-df_yy,  df_xy], dim=-1)[...,None]
        # # K2 = torch.cat([ df_yx, -df_xx], dim=-1)[...,None]
        # # by row
        # u = -df_yy + df_xy
        # v =  df_xy - df_xx
        
        # uv = torch.cat([u, v], dim=-1)
        # assert uv.shape[1] == 2
        # uv = torch.tanh(uv)

        f_dxy = torch.autograd.grad(f, xy, torch.ones_like(f),
                        create_graph=True, 
                        retain_graph=True,
                        only_inputs=True)[0]
        uv = torch.cat([f_dxy[:,1:], -f_dxy[:,:1]], dim=-1)
        return uv

    
class DivFreeRFFNet(nn.Module):
    
    def __init__(self, name, dim_mpl_layers, f_nfeatures, f_scale):
        super(DivFreeRFFNet, self).__init__()
        self.name = name
        
        # regression/pinn network       
        self.rff = Fourier(f_nfeatures, f_scale) # directly the random matrix 'cause of checkpoint and load
        self.mlp = MLP(dim_mpl_layers)
        self.div = DivFree()
        
    
    def forward(self, xin): # x := BxC(Batch, InputChannels)
        ## implement periodicity
        xin.requires_grad_(True)
        ## Fourier features
        x = self.rff(xin) # Batch x Fourier Features
        ## MLP
        x = self.mlp(x)
        ## DivFREE
        x = self.div(x, xin)
        return x

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
                div_u_xy = div_u_xy.sum().item()

                loss = F.mse_loss(y_hat, y_batch)
                current_loss += (1/batches) * (loss.item() - current_loss)
                loss.backward()
                optimiser.step()
                
                if epoch % 100 == 0 or epoch in [1, epochs]:
                    print('Epoch: %d, Loss: %f' % (epoch, current_loss))
                    print('  Div:', div_u_xy)

        print('Done with Training')
        print('Final error:', current_loss)