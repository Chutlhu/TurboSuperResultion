import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from turboflow.models.basics import *
from turboflow.utils import phy_utils as phy
from turboflow.utils import torch_utils as tch
from turboflow.utils import file_utils as fle
import turboflow.evaluation as evl

import matplotlib.pyplot as plt

from turboflow.models.phyrff_hard import MLP, DivFree, Fourier

class kDivFree(nn.Module):
    """
    Given a Radial Basis Function $\psi(x)$, the matrix-valued RBFs $\Phi(x)$ in which
    the vector fields defined by the columns are divergence-free is constructed as follows
            $\Phi_df(x) = (\nabla \nabla^T - \nabla^2 I)\psi(x)$ (1).
    
    In the 2D case and assuming that our MLP ~ RBFs (***STRONG ASSUMPTION HERE***), we can compute our vector field as follows.
    
    $\Phi_df(x) = ( [ d/dxx   d/dxy  ] - [d/dxx + d/dyy           0        ] ) MLP(x)
                  ( [ d/dyx   d/dyy  ] - [      0           d/dxx + d/dyy  ] )
                
                = ( [ -d/dyy    d/dxy  ]) Phi(x)
                  ( [  d/dyx   -d/dxx  ])
    
    u = -dMLP(x)/dyy + dMLP(x)/dxy
    v =  dMLP(x)/dxy - dMLP(x)/dxx
    
    
    References:
        - [Macedo et Casto,  2010](https://www.yumpu.com/en/document/read/37810994/learning-divergence-free-and-curl-free-vector-fields-with-matrix-)
        - [Colin P. McNally, 2011](https://arxiv.org/pdf/1102.4852.pdf)
    """

    def __init__(self):
        super(kDivFree, self).__init__()
        layers = [2] + [100] + [1]
        self.mlp = MLP(layers, 'tanh')

    def forward(self, f, xy):
        f = self.mlp(xy)
        
        # compute first order deriv
        df_xy = torch.autograd.grad(f, xy, torch.ones_like(f), 
                                    create_graph=True,
                                    retain_graph=True)[0]
        # separate x and y components
        df_x, df_y = df_xy.split(1,-1)
        
        # compute secord order deriv wrt to x
        df_x_xy = torch.autograd.grad(df_x, xy, torch.ones_like(df_x), 
                                      create_graph=True,
                                      retain_graph=True)[0]
        # compute secord order deriv wrt to y
        df_y_xy = torch.autograd.grad(df_y, xy, torch.ones_like(df_y),
                                      create_graph=True,
                                      retain_graph=True)[0]
        
        df_xx, df_xy = df_x_xy.split(1,-1)
        df_yx, df_yy = df_y_xy.split(1,-1)

        # the columns of K make a divergence-free field
        u =  df_xy - df_yy
        v =  df_xy - df_xx

        u = torch.cat([u,v], dim=-1)
        return u

def gaussian(xin):
    phi = torch.exp(-xin.pow(2))
    return phi



class RBF(nn.Module):
    def __init__(self, in_features, n_centers, out_features):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.n_centers = n_centers
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(n_centers, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(n_centers))
        self.alphas_uvx = nn.Parameter(torch.Tensor(n_centers, 1))
        self.alphas_uvy = nn.Parameter(torch.Tensor(n_centers, 1))
        self.basis_func = gaussian
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.centres, 0, 1)
        nn.init.uniform_(self.alphas_uvx, 0, 1)
        nn.init.uniform_(self.alphas_uvy, 0, 1)
        nn.init.normal_(self.log_sigmas, 0)

    def forward(self, x, xin):
        
        y, eps = x.split(self.n_centers,-1)
        assert y.shape == eps.shape
        
        # print(y)
        y = torch.exp(y**2) # N x C

        # print(y)
        eps = torch.softmax(eps, dim=1) # N x C
                
        res = []
        for k in range(y.shape[-1]):
            # compute first order deriv
            dy_xy = torch.autograd.grad(y[:,k], xin, torch.ones_like(y[:,k]), 
                                        create_graph=True,
                                        retain_graph=True)[0]
            dy_x, dy_y = dy_xy.split(1,-1)
            # compute secord order deriv
            dy_x_xy = torch.autograd.grad(dy_x, xin, torch.ones_like(dy_x), 
                                        create_graph=True,
                                        retain_graph=True)[0]
            dy_xx, dy_xy = dy_x_xy.split(1,-1)
            dy_y_xy = torch.autograd.grad(dy_y, xin, torch.ones_like(dy_y),
                                        create_graph=True,
                                        retain_graph=True)[0]
            dy_yx, dy_yy = dy_y_xy.split(1,-1)
    
            u =  (dy_xy - dy_yy)[:,:,None]
            v =  (dy_xy - dy_xx)[:,:,None]
            res.append(torch.cat([u,v], dim=1))

            # u = -dy_y[:,:,None]
            # v =  dy_x[:,:,None]
            # res.append(torch.cat([u,v], dim=1))

        res = torch.cat(res, dim=-1) # B x D x K
        res = torch.einsum('bdk,bk->bd', res, eps)

        return res

class plDivFreeRFFNet(pl.LightningModule):

    def __init__(self,  name:str, mlp_layers_num:int, mlp_layers_dim:int, 
                        mlp_last_actfn:str,
                        do_rff:bool, rff_num:int, rff_scale:float,
                        do_kdivfree:bool, kdivfree_epoch:int,
                        lam_pde:float, lam_div:float, lam_reg:float, lam_sfn:float, lam_spec:float, lam_weight:float,
                        sfn_min_x:float, sfn_num_centers:int, sfn_num_increments:int, sfn_patch_dim:int):

        super(plDivFreeRFFNet, self).__init__()

        self.save_hyperparameters()

        self.name = name
        self.automatic_optimization = True

        # regression/pinn network
        self.rff = Fourier(rff_num, rff_scale)
        
        self.do_kdivfree = do_kdivfree
        self.kdivfree_epoch = kdivfree_epoch
        self.mlp = MLP([2*rff_num] + 3*[256] + [2], mlp_last_actfn)
        self.rbf = DivFree()
    
        # off-grid regularization params
        self.min_l = sfn_min_x
        self.n_centers = sfn_num_centers
        self.patch_dim = sfn_patch_dim
        self.n_increments = sfn_num_increments
        self.sfun = lambda x, y : torch.mean(torch.abs(x - y).pow(2))
        self.sfun_model = 0.25*0.00275*torch.arange(self.n_increments)**2
        
        # PINN losses
        self.lam_pde = lam_pde
        self.lam_div = lam_div
        self.lam_reg = lam_reg
        self.lam_sfn = lam_sfn
        self.lam_spec = lam_spec
        self.lam_weight = lam_weight
        
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
        group.add_argument("--lam_div", type=float, default=0)
        group.add_argument("--lam_reg", type=float, default=0)
        group.add_argument("--lam_sfn", type=float, default=0)
        group.add_argument("--lam_spec", type=float, default=0)
        group.add_argument("--lam_weight", type=float, default=0)
        group.add_argument("--sfn_min_x", type=float, default=0.00784314)
        group.add_argument("--sfn_num_centers", type=int, default=5)
        group.add_argument("--sfn_patch_dim", type=int, default=32)
        group.add_argument("--sfn_num_increments", type=int, default=5)
        return parent_parser

    def forward(self, xin): # x := BxC(Batch, InputChannels)
        xin.requires_grad_(True)
        ## implement periodicity
        # x = torch.remainder(xin,1)
        ## Fourier features
        x = self.rff(xin) # Batch x Fourier Features
        x = self.mlp(x)
        x = self.div(x, xin)
        return x
    

    def _common_step(self, batch, batch_idx:int, stage:str):
        # It is independent of forward
        X_batch, y_batch = batch

        R = int(X_batch.shape[0]**0.5)

        ## FORWARD MLP
        y_hat = self.forward(X_batch)

        # DivFree regularization
        loss_pde = 0
        if self.lam_pde > 0:
            u, v = torch.split(y_hat,1,-1)
            du_xy = torch.autograd.grad(u, X_batch, torch.ones_like(u), create_graph=True)[0]       
            dv_xy = torch.autograd.grad(v, X_batch, torch.ones_like(v), create_graph=True)[0]
            div_autograd = du_xy[...,0] + dv_xy[...,1]
            loss_pde = torch.norm(div_autograd)**2

        # Sfun regularization
        loss_sfn = 0
        if self.lam_sfn > 0:
            # 2.c Sfun-based loss
            X_random = montecarlo_sampling_xcenters_xincerments(
                self.n_centers, self.n_increments, self.patch_dim, self.min_l, self.device)
            shape = X_random.shape # I x C x P x D

            y_random_hat = self.forward(X_random.view(-1,2))
            y_random_hat = y_random_hat.reshape(shape)

            # compute structure function
            Sfun2 = torch.mean((y_random_hat - y_random_hat[0,...])**2, dim=[3,1,2])
            loss_sfn = F.mse_loss(  torch.log(Sfun2+1e-20),
                                    torch.log(self.sfun_model.to(X_batch.device)+1e-20))

        # # Reconstruction loss
        loss_rec = F.mse_loss(y_hat, y_batch)

        # # Total loss
        loss = loss_rec + self.lam_pde*loss_pde + self.lam_sfn*loss_sfn

        # LOGs, PRINTs and PLOTs
        self.log(f'{stage}_loss', loss, on_epoch=True)
        self.log(f'{stage}/loss/rec',  loss_rec, on_epoch=True)
        self.log(f'{stage}/loss/pde',  loss_pde, on_epoch=True)
        self.log(f'{stage}/loss/sfn',  loss_sfn, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        # freeze some layers
        if self.current_epoch == self.kdivfree_epoch:
            for param in self.mlp.network[:2].parameters():
                param.requires_grad = False
        return self._common_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True) # needed for divergenge
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True) # needed for divergenge
        X_batch, y_batch = batch
        y_hat = self.forward(X_batch)
        eval_metrics = evl.compute_all_metrics(y_hat, y_batch, avg=True)
        self.log('test/metrics/reconstruction', eval_metrics['reconstruction'])
        self.log('test/metrics/angular_degree', eval_metrics['angular_degree'])
        self.log('test/metrics/log_err_specturm', eval_metrics['log_err_specturm'])
        loss_rec = F.mse_loss(y_hat, y_batch)
        return loss_rec

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

        # LOG IMAGES
        # self.prediction_image_adder(x_hr, u_hr, Pu_hr, 256, 'hr')
        
        # self.custom_histogram_adder()
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=self.lam_weight)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        return {"optimizer": optimizer, 
                # "lr_scheduler": scheduler,
                "monitor": "val_loss"}