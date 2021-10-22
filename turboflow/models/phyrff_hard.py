import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from turboflow.models.basics import *
from turboflow.utils import torch_utils as tch
from turboflow.utils import file_utils as fle

import matplotlib.pyplot as plt


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
        return None


    
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

    def __init__(self):
        super(DivFree, self).__init__()


    def forward(self, f, xy):
        
        # its hardcoded for 2 variables both for input and fx
        assert f.shape[1] == 1
        assert xy.shape[1] == 2
        
        f_xy = self.compute_ux(f, xy)
        div_free_uv = torch.cat([f_xy[:,1,None], 
                                -f_xy[:,0,None]], dim=-1)
        assert div_free_uv.shape[1] == 2
        assert div_free_uv.shape[0] == xy.shape[0]
        return div_free_uv


    def compute_ux(self, f, x):
        assert x.requires_grad

        assert f.shape[1] == 1 # must be a scalar function
        f_x = torch.autograd.grad(f, x, torch.ones_like(f),
                                  create_graph=True,
                                  retain_graph=True)[0]
        assert f_x.shape == x.shape
        return f_x
    

class DivFreeRFFNet(nn.Module):
    
    def __init__(self, name, 
                    dim_mpl_layers, last_activation_fun_name,
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
        self.mlp = MLP(dim_mpl_layers, last_activation_fun_name)
        self.div = DivFree()

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
        xin.requires_grad_(True)
        ## implement periodicity
        x = torch.remainder(xin,1)
        ## Fourier features
        if self.do_rff:
            x = self.rff(x) # Batch x Fourier Features
        ## MLP
        Px = self.mlp(x)
        ## DivFree
        u_df = self.div(Px, xin)
        return u_df, Px

    def fit(self, dataloader, epochs=1000):
        self.train()
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)
        epoch = 0

        device = tch.get_device()

        while epoch < epochs or loss < 1e-6:
            epoch += 1
            current_loss = 0
            batches = 0

            loss_rec = torch.zeros(1).to(device)
            loss_sfn = torch.zeros(1).to(device)
            loss_pde = torch.zeros(1).to(device)
            loss_reg = torch.zeros(1).to(device)

            for x_batch, y_batch in dataloader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                batches += 1
                optimiser.zero_grad()
                
                ## FORWARD
                y_hat, potential_hat = self.forward(x_batch)

                ## REGULARIZATION and LOSSES
                # L2 regularization on Potential
                if self.lam_reg > 0:
                    loss_reg = (1/batches)*torch.norm(potential_hat)**2

                # Sfun regularization
                if self.lam_sfn > 0:
                    # 2.c Sfun-based loss
                    x = self.make_offgrid_patches_xcenter_xincrement(x_batch.device)
                    I, C, P, P, D = x.shape
                    y_patches_hat = self.forward(x.reshape(I*C*P*P,D))[0].reshape(I, C, P, P, D)
                    # compute structure function
                    Sfun2 = torch.mean((y_patches_hat - y_patches_hat[0,...])**2, dim=[1,2,3,4])
                    p = 1
                    loss_sfn = torch.sum(torch.abs(torch.log(Sfun2+1e-10) - torch.log(self.sfun_model.to(x_batch.device)+1e-10))**p)

                # DivFree regularization
                if self.lam_pde > 0:
                    u, v = torch.split(y_hat,1,-1)
                    du_xy = torch.autograd.grad(u, x_batch, torch.ones_like(u), create_graph=True)[0]       
                    dv_xy = torch.autograd.grad(v, x_batch, torch.ones_like(v), create_graph=True)[0]
                    div_u_xy = du_xy[...,0] + dv_xy[...,1]
                    loss_pde = (1/batches)*torch.norm(div_u_xy)**2
            
                # Reconstruction loss
                loss_rec = (1/batches)*F.mse_loss(y_hat, y_batch)

                # Total loss
                loss = loss_rec + self.lam_pde*loss_pde + self.lam_reg*loss_reg + self.lam_sfn*loss_sfn
                current_loss +=  loss.item() - current_loss

                ## BACKWARD
                loss.backward()
                optimiser.step()

                # LOG and PRINTS
                # Logging
                self.loss_rec.append(loss_rec.item())
                self.loss_pde.append(self.lam_pde*loss_pde.item())
                self.loss_reg.append(self.lam_reg*loss_reg.item())
                self.loss_sfn.append(self.lam_sfn*loss_sfn.item())

                if self.verbose and (epoch % 100 == 0 or epoch==1):
                    print('Epoch: %4d, Loss: (rec: [%1.4f] + df: [%1.4f] + regL2: [%1.4f] + regSfun: [%1.4f]) = %f' %
                     (epoch,
                        loss_rec.item(), 
                        self.lam_pde * loss_pde.item(), 
                        self.lam_reg * loss_reg.item(), 
                        self.lam_sfn * loss_sfn.item(), 
                        current_loss))

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


class plDivFreeRFFNet(pl.LightningModule):

    def __init__(self,  name:str, mlp_layers_num:int, mlp_layers_dim:int, 
                        mlp_last_actfn:nn.Module,
                        do_rff:bool, rff_num:int, rff_scale:float,
                        do_divfree:bool,
                        lam_pde:float, lam_reg:float, lam_sfn:float,
                        sfn_min_x:float, sfn_num_centers:int, sfn_num_increments:int, sfn_patch_dim:int):

        super(plDivFreeRFFNet, self).__init__()

        self.save_hyperparameters()

        self.name = name
        self.automatic_optimization = True

        # regression/pinn network 
        if do_rff:
            self.rff = Fourier(rff_num, rff_scale) # directly the random matrix 'cause of checkpoint and load
            mlp_layers_dim = [2*rff_num] + mlp_layers_num * [mlp_layers_dim] + [1]
        else:
            self.rff = None
            mlp_layers_dim = [2]  + mlp_layers_num * [mlp_layers_dim] + [1]

        self.do_divfree = do_divfree
        if do_divfree:
            self.mlp = MLP(mlp_layers_dim, mlp_last_actfn)
            self.div = DivFree()
        else:
            mlp_layers_dim[-1] = 2
            self.mlp = MLP(mlp_layers_dim, mlp_last_actfn)

        # off-grid regularization params
        self.min_l = sfn_min_x
        self.n_centers = sfn_num_centers
        self.patch_dim = sfn_patch_dim
        self.n_increments = sfn_num_increments
        self.sfun = lambda x, y : torch.mean(torch.abs(x - y).pow(2))
        self.sfun_model = 0.25*0.00275*torch.arange(self.n_increments)**2
        
        # PINN losses
        self.lam_pde = lam_pde
        self.lam_reg = lam_reg
        self.lam_sfn = lam_sfn
        
        # reference image for logging on tensorboard
        patch_ln = torch.linspace(0, 1, 64)
        patch_sq = torch.stack(torch.meshgrid(patch_ln, patch_ln), dim=-1)
        self.reference_input_lr = patch_sq.view(-1,2)
        patch_ln = torch.linspace(0, 1, 256)
        patch_sq = torch.stack(torch.meshgrid(patch_ln, patch_ln), dim=-1)
        self.reference_input_hr = patch_sq.view(-1,2)

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("model")
        group.add_argument("--name", type=str, default="DivFreeRFFNet")
        group.add_argument("--mlp_layers_num", type=int, default=3)
        group.add_argument("--mlp_layers_dim", type=int, default=256)
        group.add_argument("--mlp_last_actfn", type=str, default="tanh")
        group.add_argument("--do_rff",     type=fle.str2bool, nargs='?', const=True, default=False)
        group.add_argument("--do_divfree", type=fle.str2bool, nargs='?', const=True, default=False)
        group.add_argument("--rff_num", type=int, default=256)
        group.add_argument("--rff_scale", type=float, default=10)
        group.add_argument("--lam_pde", type=float, default=0)
        group.add_argument("--lam_reg", type=float, default=0)
        group.add_argument("--lam_sfn", type=float, default=0)
        group.add_argument("--sfn_min_x", type=float, default=0.00784314)
        group.add_argument("--sfn_num_centers", type=int, default=5)
        group.add_argument("--sfn_patch_dim", type=int, default=32)
        group.add_argument("--sfn_num_increments", type=int, default=5)
        return parent_parser

    def forward(self, xin): # x := BxC(Batch, InputChannels)
        xin.requires_grad_(True)
        ## implement periodicity
        x = torch.remainder(xin,1)
        ## Fourier features
        if self.hparams.do_rff:
            x = self.rff(x) # Batch x Fourier Features
        
        ## MLP
        if self.hparams.do_divfree:
            Px = self.mlp(x)
            ## DivFree
            u_df = self.div(Px, xin)
            return u_df, Px
        else:
            return self.mlp(x), None

    def _common_step(self, batch, batch_idx:int, stage:str):
        # It is independent of forward
        X_batch, y_batch = batch

        # ## FORWARD
        y_hat, Py_hat = self.forward(X_batch)

        ## REGULARIZATION and LOSSES
        # L2 regularization on the gradient of the potential
        loss_reg = 0
        if self.lam_reg > 0:
            dP_xy = torch.autograd.grad(Py_hat, X_batch, torch.ones_like(Py_hat), create_graph=True)[0]       
            loss_reg = torch.norm(dP_xy[:,0] + dP_xy[:,1])**2

        # Sfun regularization
        loss_sfn = 0
        if self.lam_sfn > 0:
            # 2.c Sfun-based loss
            # X_patches = make_offgrid_patches_xcenter_xincrement(
            #     self.n_increments, self.n_centers, self.min_l, self.patch_dim, self.device)
            # I, C, P, P, D = X_patches.shape
            # y_patches_hat, _ = self.forward(X_patches.reshape(I*C*P*P,D))
            # y_patches_hat = y_patches_hat.reshape(I, C, P, P, D)
            X_random = montecarlo_sampling_xcenters_xincerments(
                self.n_centers, self.n_increments, self.patch_dim, self.min_l, self.device)
            shape = X_random.shape # I x C x P x D

            y_random_hat, _ = self.forward(X_random.view(-1,2))
            y_random_hat = y_random_hat.reshape(shape)

            # compute structure function
            Sfun2 = torch.mean((y_random_hat - y_random_hat[0,...])**2, dim=[3,1,2])
            loss_sfn = torch.sum(torch.abs(torch.log(Sfun2+1e-10) - torch.log(self.sfun_model.to(X_batch.device)+1e-10))**2)

            if self.current_epoch % 100 == 0:
                plt.loglog(Sfun2.detach().cpu().numpy())
                plt.loglog(self.sfun_model.detach().cpu().numpy())
                plt.savefig(f'./sfun_epoch-{self.current_epoch}.png')
                plt.close()

        # DivFree regularization
        loss_pde = 0
        if self.lam_pde > 0:
            u, v = torch.split(y_hat,1,-1)
            du_xy = torch.autograd.grad(u, X_batch, torch.ones_like(u), create_graph=True)[0]       
            dv_xy = torch.autograd.grad(v, X_batch, torch.ones_like(v), create_graph=True)[0]
            loss_pde = torch.norm(du_xy[...,0] + dv_xy[...,1])**2
    
        # # Reconstruction loss
        loss_rec = F.mse_loss(y_hat, y_batch)

        # # Total loss
        loss = loss_rec + self.lam_pde*loss_pde + self.lam_reg*loss_reg + self.lam_sfn*loss_sfn
        
        # LOGs, PRINTs and PLOTs
        self.log(f'{stage}_loss', loss, on_epoch=True)
        self.log(f'{stage}_sfn', loss_sfn, on_epoch=True)
        self.log(f'{stage}_rec', loss_rec, on_epoch=True)
        self.log(f'{stage}_pde', loss_pde, on_epoch=True)
        self.log(f'{stage}_reg', loss_reg, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True) # needed for divergenge
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True) # needed for divergenge
        return self._common_step(batch, batch_idx, 'test')

    #  add optional logic at the end of the training
    #  the function is called after every epoch is completed
    def training_epoch_end(self, outputs):
        return None

    def prediction_image_adder(self,x,u,Pu,R,res):
        x = x.reshape(R,R,2)
        u = u.reshape(R,R,2)
        Pu = Pu.reshape(R,R)
        self.logger.experiment.add_image(f"input/{res}/x", torch.Tensor.cpu(x[:,:,0]),
                                         self.current_epoch,dataformats="HW")
        self.logger.experiment.add_image(f"input/{res}/y", torch.Tensor.cpu(x[:,:,1]),
                                         self.current_epoch,dataformats="HW")
        self.logger.experiment.add_image(f"output/{res}/ux", torch.Tensor.cpu(u[:,:,0]),
                                         self.current_epoch,dataformats="HW")
        self.logger.experiment.add_image(f"output/{res}/uy", torch.Tensor.cpu(u[:,:,1]),
                                         self.current_epoch,dataformats="HW")
        self.logger.experiment.add_image(f"output/{res}/Pu", torch.Tensor.cpu(Pu),
                                         self.current_epoch,dataformats="HW")

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def validation_epoch_end(self, outputs):
        # ## Save Model graph
        # dummy_input = torch.rand((3,2))
        # dummy_input.requires_grad_(True)
        # self.logger.experiment.add_graph(plDivFreeRFFNet(self.hparams), dummy_input)
        # 1/0

        # LOG IMAGES
        # x_lr = self.reference_input_lr.clone().to(self.device)
        # u_lr, Pu_lr = self.forward(x_lr)
        # self.prediction_image_adder(x_lr, u_lr, Pu_lr, 64, 'lr')
        # x_hr = self.reference_input_hr.clone().to(self.device)
        # u_hr, Pu_hr = self.forward(x_hr)
        # self.prediction_image_adder(x_hr, u_hr, Pu_hr, 256, 'hr')
        
        # self.custom_histogram_adder()
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        return {"optimizer": optimizer, 
                # "lr_scheduler": scheduler,
                "monitor": "val_loss"}