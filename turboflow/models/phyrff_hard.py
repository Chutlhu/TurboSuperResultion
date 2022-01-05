import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from turboflow.models.basics import *
from turboflow.utils import phy_utils as phy
from turboflow.utils import torch_utils as tch
from turboflow.utils import file_utils as fle
import turboflow.evaluation as evl

import matplotlib.pyplot as plt

from kornia.filters import SpatialGradient, Laplacian

root_dir = '/home/dicarlo_d/Documents/Code/TurboSuperResultion/'

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
        if fun_name == 'id':
            return nn.Identity()
        if fun_name == 'ELU':
            return nn.ELU()
        return None

class MTL(nn.Module):

    def __init__(self):
        super(MTL, self).__init__()

        self.log_sigma_sqr_rec = nn.Parameter(torch.Tensor([2])).float()
        self.log_sigma_sqr_sdiv = nn.Parameter(torch.Tensor([2])).float()
        self.log_sigma_sqr_sfn = nn.Parameter(torch.Tensor([2])).float()
        self.log_sigma_sqr_grads = nn.Parameter(torch.Tensor([2])).float()

    def forward(self, x):
        return x
        
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
    


class plDivFreeRFFNet(pl.LightningModule):

    def __init__(self,  name:str, mlp_layers_num:int, mlp_layers_dim:int, 
                        mlp_last_actfn:str,
                        do_rff:bool, rff_num:int, rff_scale:float,
                        do_divfree:bool,
                        lam_pde:float, 
                        lam_sdiv:float,
                        lam_sfn:float,
                        lam_spec:float,
                        lam_curl:float,
                        lam_grads:float, lam_weight:float,
                        sfn_min_x:float, sfn_num_centers:int, sfn_num_increments:int, sfn_patch_dim:int):

        super(plDivFreeRFFNet, self).__init__()

        self.save_hyperparameters()

        self.name = name
        self.automatic_optimization = True

        # regression/pinn network
        if rff_num == 0:
            do_rff = False
        self.do_rff = rff_num
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
        
        # self.mtl = MTL()

        # off-grid regularization params
        self.min_l = sfn_min_x
        self.n_centers = sfn_num_centers
        self.patch_dim = sfn_patch_dim
        self.n_increments = sfn_num_increments
        self.sfun = lambda x, y : torch.mean(torch.abs(x - y).pow(2))
        self.sfun_model = 0.25*0.00275*torch.arange(self.n_increments)**2
        self.sfun_eps = 1e-20
        self.spec_model = torch.load(root_dir + 'data/hr_spect256.pt')
        
        self.sp_grad = SpatialGradient(mode='diff', order=1, normalized=True)
        self.sp_lapl = Laplacian(kernel_size=5, normalized=True)
        
        # PINN losses
        self.lam_pde = lam_pde
        self.lam_sdiv = lam_sdiv
        self.lam_sfn = lam_sfn
        self.lam_spec = lam_spec
        self.lam_grads = lam_grads
        self.lam_curl = lam_curl
        self.lam_weight = lam_weight

        
        # reference image for logging on tensorboard
        self.L = 64
        patch_ln_lr = torch.linspace(0, 1, self.L)
        patch_sq_lr = torch.stack(torch.meshgrid(patch_ln_lr, patch_ln_lr), dim=-1)
        self.reference_input_lr = patch_sq_lr.view(-1,2)
        self.H = 256
        patch_ln_hr = torch.linspace(0, 1, self.H)
        patch_sq_hr = torch.stack(torch.meshgrid(patch_ln_hr, patch_ln_hr), dim=-1)
        self.patch_sq_hr = patch_sq_hr
        self.reference_input_hr = patch_sq_hr.view(-1,2)

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
        
        group.add_argument("--lam_pde", type=float, default=0)      # 
        group.add_argument("--lam_sdiv", type=float, default=0)     # Soft constrain at full resolution
        # group.add_argument("--lam_reg", type=float, default=0)    # deprecated.
        group.add_argument("--lam_curl", type=float, default=0)     # || grad w ||_2^2
        group.add_argument("--lam_spec", type=float, default=0)     # Target spectrogram at full resolution
        group.add_argument("--lam_grads", type=float, default=0)    # Autograd = Spatial Gradient at Full-Resolution
        group.add_argument("--lam_weight", type=float, default=0)   # L2 regularization on the NN's weights (directly in the torch optimizer)
        
        group.add_argument("--lam_sfn", type=float, default=0)
        group.add_argument("--sfn_min_x", type=float, default=1./256.)
        group.add_argument("--sfn_num_centers", type=int, default=5)
        group.add_argument("--sfn_patch_dim", type=int, default=32)
        group.add_argument("--sfn_num_increments", type=int, default=5)
        return parent_parser

    def forward(self, xin): # x := BxC(Batch, InputChannels)
        xin.requires_grad_(True)
        ## implement periodicity
        x = torch.remainder(xin,1)
        # x = 2*torch.abs(xin/2 - torch.floor(xin/2 + 0.5))
        ## Fourier features
        if self.do_rff:
            x = self.rff(x) # Batch x Fourier Features
        
        ## MLP
        if self.do_divfree:
            Px = self.mlp(x)
            ## DivFree
            u_df = self.div(Px, xin)
            return u_df, Px
        else:
            x = self.mlp(x)
            return x, None

    def _common_step(self, batch, batch_idx:int, stage:str):
        # It is independent of forward
        X_batch, y_batch = batch

        B = X_batch.shape[0]
        R = int(B*0.5)

        ## FORWARD BATCH
        y_hat, Py_hat = self.forward(X_batch)

        # # Reconstruction loss
        loss_rec = F.mse_loss(y_hat, y_batch)

        ## REGULARIZATION and LOSSES
        # OFF GRID REGULARIZATIAN on FULL GRID
        ## FORWAND OFFGRID
        # if self.lam_sdiv > 0:
        #     x_off = make_offgrid_patches_xcenter(self.n_centers, self.min_l, patch_dim=self.patch_dim, device=X_batch.device)

        #     shape = x_off.shape # C x P x P x D
        #     N, P, p, D = shape
        #     x_off = x_off.view(-1, 2) # Bx2
        #     x_off.requires_grad_(True)
        #     y_hat_off, _ = self.forward(x_off) #Bx2

        #     u, v = torch.split(y_hat_off,1,-1)
        #     du_xy = torch.autograd.grad(u, x_off, torch.ones_like(u), create_graph=True)[0] # Bx2 
        #     dv_xy = torch.autograd.grad(v, x_off, torch.ones_like(v), create_graph=True)[0] # Bx2


        # Full Resolution spectrum
        loss_spec = 0
        if self.lam_spec > 0:
            # tke_spec_gt = phy.energy_spectrum(y_batch.reshape(self.H,self.H,2).permute(2,0,1))[0]
            # tke_spec_gt = self.spec_model.to(X_batch.device)
            tke_spec_gt = phy.energy_spectrum(y_batch.reshape(self.L,self.L,2).permute(2,0,1))[0]
            y_hat_lr = self.forward(self.reference_input_lr.to(X_batch.device))[0]
            tke_spec_batch = phy.energy_spectrum(y_hat_lr.reshape(self.L,self.L,2).permute(2,0,1))[0]
            tke_log_error = torch.sum(torch.abs(tke_spec_gt - tke_spec_batch))

            # tke_spec_batch = phy.energy_spectrum(y_hat_off.reshape(self.H, self.H,2).permute(2,0,1))[0]
            loss_spec = torch.sum(tke_log_error)
  
        my_autograd = lambda y, x : torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

        if self.lam_grads + self.lam_sdiv + self.lam_sdiv + self.lam_curl + self.lam_pde > 0:
            X_hr = self.reference_input_hr.to(X_batch.device)
            P = self.H
            X_hr.requires_grad_(True)
            y_hat_hr = self.forward(X_hr)[0]
            du_xy = my_autograd(y_hat_hr[:,0], X_hr) # Bx2 
            dv_xy = my_autograd(y_hat_hr[:,1], X_hr) # Bx2

        # Spatial gradient at very high resolution ~ Autograd
        loss_grads = 0
        if self.lam_grads > 0:
            N = 1
            y_img_hat = y_hat_hr.view(P,P,2)[None,...].permute(0,3,1,2) # BxC -> NxPxPx2 -> Nx2xHxW

            # compute the spatial gradient
            # y_img_hat = y_hat_off.view(shape).permute(0,3,1,2) # BxC -> NxPxPx2 -> Nx2xHxW
            grads = self.sp_grad(y_img_hat) # NxCx2xHxW

            gdiff_u_x = F.mse_loss(du_xy[:,0].reshape(N,P,P) / P, grads[:,0,1,...])
            gdiff_u_y = F.mse_loss(du_xy[:,1].reshape(N,P,P) / P, grads[:,0,0,...])
            gdiff_v_x = F.mse_loss(dv_xy[:,0].reshape(N,P,P) / P, grads[:,1,1,...])
            gdiff_v_y = F.mse_loss(dv_xy[:,1].reshape(N,P,P) / P, grads[:,1,0,...])

            loss_grads = (gdiff_u_x + gdiff_u_y + gdiff_v_x + gdiff_v_y) / 4

        # Soft DivFree regularization at high resolution
        loss_sdiv = 0
        if self.lam_sdiv > 0:
            div_autograd = du_xy[:,0] + dv_xy[:,1]
            loss_sdiv = torch.mean(div_autograd**2)

        if self.lam_curl + self.lam_pde > 0:
            w = dv_xy[:,0] - du_xy[:,1]
            dw_xy = my_autograd(w, X_hr) # Bx2
            
        loss_curl = 0
        if self.lam_curl > 0:
            loss_curl = torch.mean(torch.norm(dw_xy, dim=1)**2)

        loss_pde = 0
        if self.lam_pde > 0:
            left = w*(y_hat_hr[:,0] + y_hat_hr[:,1])
            dw_dxy = self.sp_grad(w.reshape(P,P)[None,None,...]) # BxCx2xHxW
            d2w_dxy = self.sp_grad(dw_dxy[0,...]) # BxCx2xHxW
            lapl_w = d2w_dxy[:,0,1,:,:] + d2w_dxy[:,1,0,:,:]
            right = (-(1./3000.)*lapl_w).flatten()
            loss_pde = torch.mean(torch.abs(left - right)**2)

            if self.current_epoch % 10 == 0:
                plt.figure(figsize=(15,5))
                plt.suptitle(f'Loss: {loss_pde.item()}')
                plt.subplot(131)
                plt.imshow(left.reshape(self.H, self.H).detach().cpu())
                plt.colorbar()
                plt.subplot(132)
                plt.imshow(right.reshape(self.H, self.H).detach().cpu())
                plt.colorbar()
                plt.subplot(133)
                plt.imshow(torch.abs(left-right).reshape(self.H, self.H).detach().cpu())
                plt.colorbar()
                plt.show()

            del right
            del left

        # OFF GRID REGULARIZATIAN on SPARSE GRID (MONTECARLO Sampling)
        # L2 regularization on the gradient of the potential
        if self.lam_sfn > 0:
            # X_random = montecarlo_sampling_xcenters_xincerments(
            #     self.n_centers, self.n_increments, self.patch_dim, self.min_l, self.device)
            # self.reference_input_hr = patch_sq.view(-1,2)
            # shape = X_random.shape # I x C x P x D
            # X_random.requires_grad_(True)
            # self.H = 256
            # patch_ln = torch.linspace(0, 1, self.H)
            # patch_sq = torch.meshgrid(patch_ln, patch_ln, device=X_batch.device) # H x H
            patch_sq = self.patch_sq_hr.to(X_batch.device) # H x H x 2
            size = (self.n_increments, *patch_sq.shape) 
            X_incr = patch_sq.unsqueeze(0).expand(size) # I x H x H x D
            increments = self.min_l * torch.arange(0,self.n_increments,device=X_batch.device) 
            X_incr = X_incr + increments[:,None,None,None] 
            
            shape_incr = X_incr.shape
            y_incr = self.forward(X_incr.view(-1,2))[0]
        
        # Sfun regularization
        loss_sfn = 0
        if self.lam_sfn > 0:
            
            y_incr_hat = y_incr.reshape(shape_incr)
            # compute structure function
            Sfun2 = torch.mean((y_incr_hat - y_incr_hat[0,...])**2, dim=[3,1,2])
            loss_sfn = F.mse_loss(  torch.log(Sfun2+self.sfun_eps),
                                    torch.log(self.sfun_model.to(X_batch.device)+self.sfun_eps))


            if stage == 'val':
                plt.plot(Sfun2.detach().cpu() + self.sfun_eps)
                plt.plot(self.sfun_model.detach().cpu()+self.sfun_eps)
                plt.show()
            
            del y_incr
            del X_incr
            del patch_sq

        loss = loss_rec \
            + self.lam_sfn * loss_sfn \
            + self.lam_grads * loss_grads \
            + self.lam_sdiv * loss_sdiv \
            + self.lam_spec * loss_spec \
            + self.lam_curl * loss_curl \
            + self.lam_pde * loss_pde
        
        # LOGs, PRINTs and PLOTs
        self.log(f'{stage}/loss/tot',  loss,       on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/rec',  loss_rec,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/sdiv', loss_sdiv,  on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/sfn',  loss_sfn,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/grad', loss_grads, on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/spec', loss_spec,  on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/curl', loss_curl,  on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/pde',  loss_pde,   on_epoch=True, on_step=False)


        err_rec = evl.recostruction_error(y_hat, y_batch, avg=True)
        err_ang = evl.angular_error_2Dfield(y_hat, y_batch, avg=True)
        self.log(f'{stage}/metrics/reconstruction', err_rec)
        self.log(f'{stage}/metrics/angular_degree', err_ang)
        # self.log(f'{stage}/metrics/log_err_specturm', eval_metrics['log_err_specturm'])

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True) # needed for divergenge
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True) # needed for divergenge
        X_batch, y_batch = batch
        y_hat, Py_hat = self.forward(X_batch)

        err_rec = evl.recostruction_error(y_hat, y_batch, avg=True)
        err_ang = evl.angular_error_2Dfield(y_hat, y_batch, avg=True)
        self.log('test/metrics/reconstruction', err_rec)
        self.log('test/metrics/angular_degree', err_ang)
        # eval_metrics = evl.compute_all_metrics(y_hat, y_batch, avg=True)
        # self.log('test/metrics/reconstruction', eval_metrics['reconstruction'])
        # self.log('test/metrics/angular_degree', eval_metrics['angular_degree'])
        # self.log('test/metrics/log_err_specturm', eval_metrics['log_err_specturm'])
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.5e-3, weight_decay=self.lam_weight)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler,
                "monitor": "train/loss/tot"}


# class DivFreeRFFNet(nn.Module):
    
#     def __init__(self, name, 
#                     dim_mpl_layers, last_activation_fun_name,
#                     do_rff, f_nfeatures, f_scale, 
#                     smallest_increment, n_increments, n_centers,
#                     lam_reg=1, lam_sfn=1, lam_pde=1,
#                     verbose=True):

#         super(DivFreeRFFNet, self).__init__()
#         self.name = name
#         self.verbose = verbose

#         assert dim_mpl_layers[-1] == 1
        
#         # regression/pinn network 
#         self.do_rff = do_rff
#         if do_rff:
#             self.rff = Fourier(f_nfeatures, f_scale) # directly the random matrix 'cause of checkpoint and load
#             dim_mpl_layers[0] = dim_mpl_layers[0]*f_nfeatures
#         else:
#             self.rff = None

#         self.rff = Fourier(f_nfeatures, f_scale)
#         self.mlp = MLP(dim_mpl_layers, last_activation_fun_name)
#         self.div = DivFree()

#         # off-grid regularization params
#         self.min_l = smallest_increment
#         self.n_centers = n_centers
#         self.patch_dim = 32
#         self.n_increments = n_increments
#         self.sfun = lambda x, y : torch.mean(torch.abs(x - y).pow(2))
#         # self.sfun_model = 0.000275*smallest_increment*torch.arange(self.n_increments)**2
#         self.sfun_model = 0.25*0.00275*torch.arange(self.n_increments)**2
        
#         self.lam_pde = lam_pde
#         self.lam_reg = lam_reg
#         self.lam_sfn = lam_sfn

#         self.loss = []
#         self.loss_pde = []
#         self.loss_rec = []
#         self.loss_reg = []
#         self.loss_sfn = []
        
#     def forward(self, xin): # x := BxC(Batch, InputChannels)
#         xin.requires_grad_(True)
#         ## implement periodicity
#         x = torch.remainder(xin,1)
#         ## Fourier features
#         if self.do_rff:
#             x = self.rff(x) # Batch x Fourier Features
#         ## MLP
#         Px = self.mlp(x)
#         ## DivFree
#         u_df = self.div(Px, xin)
#         return u_df, Px

#     def fit(self, dataloader, epochs=1000):
#         self.train()
#         optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)
#         epoch = 0

#         device = tch.get_device()

#         while epoch < epochs or loss < 1e-6:
#             epoch += 1
#             current_loss = 0
#             batches = 0

#             loss_rec = torch.zeros(1).to(device)
#             loss_sfn = torch.zeros(1).to(device)
#             loss_pde = torch.zeros(1).to(device)
#             loss_reg = torch.zeros(1).to(device)

#             for x_batch, y_batch in dataloader:

#                 x_batch = x_batch.to(device)
#                 y_batch = y_batch.to(device)

#                 batches += 1
#                 optimiser.zero_grad()
                
#                 ## FORWARD
#                 y_hat, potential_hat = self.forward(x_batch)

#                 ## REGULARIZATION and LOSSES
#                 # L2 regularization on Potential
#                 if self.lam_reg > 0:
#                     loss_reg = (1/batches)*torch.norm(potential_hat)**2

#                 # Sfun regularization
#                 if self.lam_sfn > 0:
#                     # 2.c Sfun-based loss
#                     x = self.make_offgrid_patches_xcenter_xincrement(x_batch.device)
#                     I, C, P, P, D = x.shape
#                     y_patches_hat = self.forward(x.reshape(I*C*P*P,D))[0].reshape(I, C, P, P, D)
#                     # compute structure function
#                     Sfun2 = torch.mean((y_patches_hat - y_patches_hat[0,...])**2, dim=[1,2,3,4])
#                     p = 1
#                     loss_sfn = torch.sum(torch.abs(torch.log(Sfun2+1e-10) - torch.log(self.sfun_model.to(x_batch.device)+1e-10))**p)

#                 # DivFree regularization
#                 if self.lam_pde > 0:
#                     u, v = torch.split(y_hat,1,-1)
#                     du_xy = torch.autograd.grad(u, x_batch, torch.ones_like(u), create_graph=True)[0]       
#                     dv_xy = torch.autograd.grad(v, x_batch, torch.ones_like(v), create_graph=True)[0]
#                     div_u_xy = du_xy[...,0] + dv_xy[...,1]
#                     loss_pde = (1/batches)*torch.norm(div_u_xy)**2
            
#                 # Reconstruction loss
#                 loss_rec = (1/batches)*F.mse_loss(y_hat, y_batch)

#                 # Total loss
#                 loss = loss_rec + self.lam_pde*loss_pde + self.lam_reg*loss_reg + self.lam_sfn*loss_sfn
#                 current_loss +=  loss.item() - current_loss

#                 ## BACKWARD
#                 loss.backward()
#                 optimiser.step()

#                 # LOG and PRINTS
#                 # Logging
#                 self.loss_rec.append(loss_rec.item())
#                 self.loss_pde.append(self.lam_pde*loss_pde.item())
#                 self.loss_reg.append(self.lam_reg*loss_reg.item())
#                 self.loss_sfn.append(self.lam_sfn*loss_sfn.item())

#                 if self.verbose and (epoch % 100 == 0 or epoch==1):
#                     print('Epoch: %4d, Loss: (rec: [%1.4f] + df: [%1.4f] + regL2: [%1.4f] + regSfun: [%1.4f]) = %f' %
#                      (epoch,
#                         loss_rec.item(), 
#                         self.lam_pde * loss_pde.item(), 
#                         self.lam_reg * loss_reg.item(), 
#                         self.lam_sfn * loss_sfn.item(), 
#                         current_loss))

#         print('Done with Training')
#         print('Final error:', current_loss)

#     def make_offgrid_patches_xcenter(self, device, n_centers = None):
#         """
#         for each random point in the image, make a square patch
#         return: C x P x P x 2
#         """
        
#         # for earch 
#         if n_centers is None:
#             n_centers = self.n_centers
#         centers = torch.randn(n_centers,2).to(device)

#         ## make a patch
#         # define one axis
#         patch_ln = torch.arange(-self.min_l*self.patch_dim, self.min_l*self.patch_dim, self.min_l, device=device)
#         # make it square meshgrid
#         patch_sq = torch.stack(torch.meshgrid(patch_ln, patch_ln), dim=-1)
        
#         ## center the patch for all the centers
#         size = (n_centers, *patch_sq.shape)
#         patch_sq_xcenter = patch_sq.unsqueeze(0).expand(size)
#         assert torch.allclose(patch_sq_xcenter[0,:,:], patch_sq)
#         assert torch.allclose(patch_sq_xcenter[3,:,:], patch_sq)
#         patch_sq_xcenter = patch_sq_xcenter + centers[:,None,None,:]
#         # some checks
#         assert len(patch_sq_xcenter.shape) == 4
#         assert patch_sq_xcenter.shape[-1] == 2
#         assert patch_sq_xcenter.shape[0] == n_centers
#         assert patch_sq_xcenter.shape[1] == patch_sq_xcenter.shape[2] == self.patch_dim*2
#         return patch_sq_xcenter

#     def make_offgrid_patches_xcenter_xincrement(self, device):
#         patches_xcenter = self.make_offgrid_patches_xcenter(device) # C x P x P x 2
#         increments = self.min_l * torch.arange(0,self.n_increments,device=patches_xcenter.device)
        
#         # expand patches for each increments
#         size = (self.n_increments, *patches_xcenter.shape)
#         patches_xcenter_xincrement = patches_xcenter.unsqueeze(0).expand(size)
#         assert torch.allclose(patches_xcenter_xincrement[0,:,:], patches_xcenter)
#         assert torch.allclose(patches_xcenter_xincrement[1,:,:], patches_xcenter)
#         patches_xcenter_xincrement = patches_xcenter_xincrement + increments[:,None,None,None,None]
#         # some checks
#         assert len(patches_xcenter_xincrement.shape) == 5
#         assert patches_xcenter_xincrement.shape[-1] == 2
#         assert patches_xcenter_xincrement.shape[0] == self.n_increments
#         assert patches_xcenter_xincrement.shape[1] == self.n_centers
#         assert patches_xcenter_xincrement.shape[2] == patches_xcenter_xincrement.shape[3] == self.patch_dim*2
#         return patches_xcenter_xincrement

