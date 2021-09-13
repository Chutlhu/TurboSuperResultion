import torch
import torch.nn as nn
import torch.functional as F

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

    
class RFFNet_pl(pl.LightningModule):
    
    def __init__(self, name, dim_mpl_layers, f_nfeatures, f_scale, random_matrix=None, lam_pde=1e-4):
        super(RFFNet_pl, self).__init__()
        self.name = name
        self.automatic_optimization = True
        
        # regression/pinn network       
        self.rff = Fourier(f_nfeatures, f_scale) # directly the random matrix 'cause of checkpoint and load
        self.mlp = MLP(dim_mpl_layers)
        
        # PINN losses
        self.lam_pde = lam_pde
        
        # other params
        # self.mode = mode # pre_train with adam, fine_tune with bsgf
        
        # for loading from checkpoints
        self.save_hyperparameters('name', 'dim_mpl_layers', 'f_nfeatures', 'f_scale', 'lam_pde')
    
    def forward(self, x): # x := BxC(Batch, InputChannels)
        ## implement periodicity
        x = torch.remainder(x,1)
        ## Fourier features
        x = self.rff(x) # Batch x Fourier Features
        ## MLP
        x = self.mlp(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, x_true = batch
        x.requires_grad_(True) # for computing the grad in the loss_pde
        x_pred = self.forward(x)
        # reconstruction loss
        loss_rec = F.mse_loss(x_pred, x_true)
        # pde loss: div(u) = 0
        if self.lam_pde > 0:
            u, v = torch.split(x_pred,1,-1)
            du_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]       
            dv_y = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
            div_u = du_x[...,0] + dv_y[...,1]
            loss_pde = torch.norm(div_u)
            loss = loss_rec + self.lam_pde*loss_pde
        else:
            loss = loss_rec
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        self.log('train_loss_data', loss_rec)
        return loss

    
    def validation_step(self, batch, batch_idx):
        # enable gradient
        torch.set_grad_enabled(True)
        x, x_true = batch
        x.requires_grad_(True) # for computing the grad in the loss_pde
        x_pred = self.forward(x)
        # reconstruction loss
        loss_rec = F.mse_loss(x_pred, x_true)
        # pde loss: div(u) = 0
        if self.lam_pde > 0:
            u, v = torch.split(x_pred,1,-1)
            du_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]       
            dv_y = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
            div_u = du_x[...,0] + dv_y[...,1]
            loss_pde = torch.norm(div_u)
            loss = loss_rec + self.lam_pde*loss_pde
        else:
            loss = loss_rec
        self.log('valid_loss', loss, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer}
#                 "lr_scheduler": scheduler, 
#                 "monitor": "valid_loss"}