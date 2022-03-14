import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from turboflow.models.core import LinearTanh, Fourier, Fourier2, TimeFourier, MLP, CNN, gMLP
from turboflow.utils import phy_utils as phy
from turboflow.utils import torch_utils as tch
from turboflow.utils import file_utils as fle
import turboflow.evaluation as evl

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kornia.filters import SpatialGradient, Laplacian

root_dir = '/home/dicarlo_d/Documents/Code/TurboSuperResultion/'
        
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
        
        # f_xy = self.compute_ux(f, xy)
        f_xy = tch.diff(f, xy)
    
        div_free_uv = torch.cat([f_xy[:,1,None], 
                                -f_xy[:,0,None]], dim=-1)
        assert div_free_uv.shape[1] == 2
        assert div_free_uv.shape[0] == xy.shape[0]
        return div_free_uv


    # def compute_ux(self, f, x):
    #     assert x.requires_grad

    #     assert f.shape[1] == 1 # must be a scalar function
    #     f_x = torch.autograd.grad(f, x, torch.ones_like(f))[0]
    #     assert f_x.shape == x.shape
    #     return f_x
    

class plDivFreeRFFNet(pl.LightningModule):

    def __init__(self,  name:str,
                        do_time:bool,
                        do_cnn:bool,
                        mlp_layers_num:int, mlp_layers_dim:int, 
                        mlp_last_actfn:str,
                        rff_num_space:int, rff_scale_space:float,
                        rff_num_time:int, rff_scale_time:float,
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

        if do_time:
            nvarin = 3
        else:
            nvarin = 2

        # regression/pinn network
        self.do_rff = True

        # space subnet
        self.rff_time = Fourier(rff_num_time, rff_scale_time, nvars=1)
        self.rff_space = Fourier(rff_num_space, rff_scale_space, nvars=2)
        
        self.do_cnn = do_cnn
        if self.do_cnn:
            if name == 'CNN':
                self.cnn = CNN(rff_num_space, rff_num_time)
                mlp_layers_dim = [256] + mlp_layers_num*[mlp_layers_dim] + [2]
                self.mlp = MLP(mlp_layers_dim, mlp_last_actfn)
            if name == 'SGU':
                # mlp_layers_dim_x = [2*rff_num_space] + [128]
                # self.mlp_x = MLP(mlp_layers_dim_x, mlp_last_actfn)

                # mlp_layers_dim_t = [2*rff_num_time] + [128]
                # self.mlp_t = MLP(mlp_layers_dim_t, mlp_last_actfn)

                self.cnn = gMLP(
                    d_model=2*rff_num_space, 
                    d_ffn=128, 
                    seq_len=2*rff_num_time, num_layers=3)
                self.sgu = self.cnn
                mlp_layers_dim = [2*rff_num_space] + mlp_layers_num*[mlp_layers_dim] + [2]
                self.mlp = MLP(mlp_layers_dim, mlp_last_actfn)
            
        else:
            mlp_layers_dim_x = [2*rff_num_space] + [128]
            self.mlp_x = MLP(mlp_layers_dim_x, mlp_last_actfn)

            mlp_layers_dim_t = [2*rff_num_time] + [128]
            self.mlp_t = MLP(mlp_layers_dim_t, mlp_last_actfn)

            mlp_layers_dim = [128] + mlp_layers_num*[mlp_layers_dim] + [16]
            self.mlp_xt = MLP(mlp_layers_dim, mlp_last_actfn)

            mlp_layers_dim = [16] + [2]
            self.mlp = MLP(mlp_layers_dim, mlp_last_actfn)
    

        # off-grid regularization params
        self.min_l = sfn_min_x
        self.n_centers = sfn_num_centers
        self.patch_dim = sfn_patch_dim
        self.n_increments = sfn_num_increments
        self.sfun = lambda x, y : torch.mean(torch.abs(x - y).pow(2))
        self.sfun_model = 0.25*0.00275*torch.arange(self.n_increments)**2
        self.sfun_eps = 1e-20
        # self.spec_model = torch.load(root_dir + 'data/hr_spect256.pt')
        
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
        self.lam_vort_ic = 0
        
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
        
        group.add_argument("--do_time",     type=fle.str2bool, nargs='?', const=True, default=False)
        
        group.add_argument("--do_cnn",     type=fle.str2bool, nargs='?', const=True, default=False)
        group.add_argument("--mlp_layers_num", type=int, default=3)
        group.add_argument("--mlp_layers_dim", type=int, default=256)
        group.add_argument("--mlp_last_actfn", type=str, default="tanh")

        group.add_argument("--do_divfree", type=fle.str2bool, nargs='?', const=True, default=False)
        group.add_argument("--rff_num", type=int, default=256)
        group.add_argument("--rff_scale", type=float, default=10)
        
        group.add_argument("--lam_pde", type=float, default=0)      # 
        group.add_argument("--lam_sdiv", type=float, default=0)     # Soft constrain at full resolution
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
        # xin[:,1:] = torch.remainder(xin[:,1:],1)
        x = xin.squeeze()
        ## Fourier features
        # if self.do_rff:
        t = self.rff_time(x[:,:1])
        x = self.rff_space(x[:,1:])

        if self.do_cnn:
            if self.name == 'CNN':
                tx = torch.matmul(t[:,:,None], x[:,None,:])
                tx = self.cnn(tx[:,None,:,:])
            elif self.name == 'SGU':
                # x = self.mlp_x(x)
                # t = self.mlp_t(t)
                tx = torch.matmul(t[:,:,None], x[:,None,:])
                tx = self.sgu(tx)
                tx = torch.mean(tx, dim=1) # lum over channels
                # xt = torch.flatten(xt, start_dim=1)
            u = self.mlp(tx)
        else:
            if self.name == 'myMLP':
                x = self.mlp_x(x)
                t = self.mlp_t(t)
                x = self.mlp_xt(x*t)
                u = self.mlp(x)
            else:
                x = self.mlp_x(x)
                t = self.mlp_t(t)
                x = self.mlp_xt(x)
                t = self.mlp_xt(t)
                u = self.mlp(x*t)
        return u, None

        ## MLP
        # if self.do_divfree:
        #     Px = self.mlp(x)
        #     ## DivFree
        #     u_df = self.div(Px, xin)
        #     return u_df, Px
        # else:
        #     x = self.mlp(x)
        #     return x, None

    def _common_step(self, batch, batch_idx:int, stage:str):
        # It is independent of forward
        X_batch, y_batch = batch

        B = X_batch.shape[0]
        H = self.H

        ## FORWARD BATCH
        y_hat, Py_hat = self.forward(X_batch)

        ## REGULARIZATION and LOSSES
        # Reconstruction loss
        loss_rec = F.mse_loss(y_hat, y_batch)

        loss_pde = 0
        loss_sdiv = 0
        loss_grads = 0
        loss_vort_ic = 0

        if self.lam_sdiv > 0:
            u, v = torch.split(y_hat, 1, -1)
            du = tch.diff(u, X_batch)
            u_t, u_x, u_y = du.split(1, -1)
            dv = tch.diff(v, X_batch)
            v_t, v_x, v_y = dv.split(1, -1)

            # Divergenge Free
            res = u_x + v_y
            loss_sdiv = F.mse_loss(res, torch.zeros_like(res))

        # w_auto = v_x - u_y
        
        # # # initial condition (=> t = 0)
        # t = 0
        # P = self.H
        # X_hr = self.reference_input_hr.to(X_batch.device).clone()
        # X_hr = torch.cat([t*torch.ones_like(X_hr[:,0])[:,None], X_hr], axis=-1)
        # X_hr.requires_grad_(True)

        # y_hr_hat, _ = self.forward(X_hr)
        # y_img_hat = y_hr_hat.view(P,P,2)[None,...].permute(0,3,1,2) # BxC -> NxPxPx2 -> Nx2xHxW
        
        # # spatial gradients
        # grads = self.sp_grad(y_img_hat) # NxCx2xHxW    
        # du_x_spat = grads[:,0,1,...]
        # du_y_spat = grads[:,0,0,...]
        # dv_x_spat = grads[:,1,1,...]
        # dv_y_spat = grads[:,1,0,...]
        # # w_spat = (dv_x_spat - du_y_spat)

        # # autograd gradients
        # u, v = torch.split(y_hr_hat, 1, -1)
        # du = tch.diff(u, X_hr)
        # u_t, u_x, u_y = du.split(1, -1)
        # dv = tch.diff(v, X_hr)
        # v_t, v_x, v_y = dv.split(1, -1)
        # # w_auto = v_x - u_y

        # gdiff_w0 = torch.norm(w_auto.reshape(1,P,P) / P - w_spat)**2
        # loss_vort_ic = gdiff_w0

        # if stage == 'val' and batch_idx == 0:
        #     plt.figure(figsize=(10,5))
        #     plt.subplot(121)
        #     plt.title('autograd')
        #     im = plt.imshow((w_auto.reshape(1,P,P) / P).squeeze().detach().cpu())
        #     plt.colorbar(im,fraction=0.046, pad=0.04)
        #     plt.subplot(122)
        #     plt.title('spatial')
        #     im = plt.imshow(w_spat.squeeze().detach().cpu())
        #     plt.colorbar(im,fraction=0.046, pad=0.04)
        #     plt.tight_layout(pad=0.05)
        #     plt.show()

        # gdiff_u_x = torch.norm(u_x.reshape(1,P,P) / P - du_x_spat)**2
        # gdiff_u_y = torch.norm(u_y.reshape(1,P,P) / P - du_y_spat)**2
        # gdiff_v_x = torch.norm(v_x.reshape(1,P,P) / P - dv_x_spat)**2
        # gdiff_v_y = torch.norm(v_y.reshape(1,P,P) / P - dv_y_spat)**2

        # loss_grads = (gdiff_u_x + gdiff_u_y + gdiff_v_x + gdiff_v_y) / 4

        # if stage == 'val':
        #     plt.figure(figsize=(20,5))
        #     plt.subplot(131)
        #     plt.title('U')
        #     im = plt.imshow((u.reshape(1,P,P)).squeeze().detach().cpu())
        #     plt.colorbar(im,fraction=0.046, pad=0.04)
        #     plt.subplot(132)
        #     plt.title('autograd')
        #     im = plt.imshow((u_x.reshape(1,P,P) / P).squeeze().detach().cpu())
        #     plt.colorbar(im,fraction=0.046, pad=0.04)
        #     plt.subplot(133)
        #     plt.title('spatial')
        #     im = plt.imshow(du_x_spat.squeeze().detach().cpu())
        #     plt.colorbar(im,fraction=0.046, pad=0.04)
        #     plt.show()


        # if self.lam_sdiv + self.lam_pde > 0:
        #     # Navier Stokes loss (vorticity equation)
        #     u, v = torch.split(y_hat, 1, -1)
            
        #     du = tch.diff(u, X_batch)
        #     u_t, u_x, u_y = du.split(1, -1)

        #     dv = tch.diff(v, X_batch)
        #     v_t, v_x, v_y = dv.split(1, -1)

            # w = v_x - u_y
            # dw = tch.diff(w, X_batch)
            # w_t, w_x, w_y = dw.split(1, -1)
            # ddw = tch.diff(dw, X_batch)
            # w_tt, w_xx, w_yy = ddw.split(1, -1)

            # # spatial change of vorticity / unit volume (u dot Nabla)w
            # w_spatial = u*w_x + v*w_y
            # # diffusion of vorticity / univ volume (nu Laplacian w)
            # w_diffusion = (1./3000.)*(w_xx + w_yy)

            # # full NS equation
            # res = w_t + w_spatial + w_diffusion
            # loss_pde = F.mse_loss(res, torch.zeros_like(res))

            # res = u_x + v_y
            # loss_sdiv = F.mse_loss(res, torch.zeros_like(res))

        # # Divergenge Free
        # res = u_x + v_y
        # loss_sdiv = F.mse_loss(res, torch.zeros_like(res))



        # # OFF GRID REGULARIZATIAN on FULL GRID
        # ## FORWAND OFFGRID

        # # if self.lam_sdiv > 0:
        # #     x_off = make_offgrid_patches_xcenter(self.n_centers, self.min_l, patch_dim=self.patch_dim, device=X_batch.device)

        # #     shape = x_off.shape # C x P x P x D
        # #     N, P, p, D = shape
        # #     x_off = x_off.view(-1, 2) # Bx2
        # #     x_off.requires_grad_(True)
        # #     y_hat_off, _ = self.forward(x_off) #Bx2

        # #     u, v = torch.split(y_hat_off,1,-1)
        # #     du_xy = torch.autograd.grad(u, x_off, torch.ones_like(u), create_graph=True)[0] # Bx2 
        # #     dv_xy = torch.autograd.grad(v, x_off, torch.ones_like(v), create_graph=True)[0] # Bx2


        # # Full Resolution spectrum
        # loss_spec = 0
        # if self.lam_spec > 0:
        #     # tke_spec_gt = phy.energy_spectrum(y_batch.reshape(self.H,self.H,2).permute(2,0,1))[0]
        #     # tke_spec_gt = self.spec_model.to(X_batch.device)
        #     tke_spec_gt = phy.energy_spectrum(y_batch.reshape(self.L,self.L,2).permute(2,0,1))[0]
        #     y_hat_lr = self.forward(self.reference_input_lr.to(X_batch.device))[0]
        #     tke_spec_batch = phy.energy_spectrum(y_hat_lr.reshape(self.L,self.L,2).permute(2,0,1))[0]
        #     tke_log_error = torch.sum(torch.abs(tke_spec_gt - tke_spec_batch))

        #     # tke_spec_batch = phy.energy_spectrum(y_hat_off.reshape(self.H, self.H,2).permute(2,0,1))[0]
        #     loss_spec = torch.sum(tke_log_error)
  
        # my_autograd = lambda y, x : torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

        # if self.lam_grads + self.lam_sdiv + self.lam_sdiv + self.lam_curl + self.lam_pde > 0:
        #     X_hr = self.reference_input_hr.to(X_batch.device)
        #     P = self.H
        #     X_hr.requires_grad_(True)
        #     y_hat_hr = self.forward(X_hr)[0]
        #     du_xy = my_autograd(y_hat_hr[:,0], X_hr) # Bx2 
        #     dv_xy = my_autograd(y_hat_hr[:,1], X_hr) # Bx2


        # # Spatial gradient at very high resolution ~ Autograd
        # loss_grads = 0
        # if self.lam_grads > 0:
        #     N = 1
        #     y_img_hat = y_hat_hr.view(P,P,2)[None,...].permute(0,3,1,2) # BxC -> NxPxPx2 -> Nx2xHxW

        #     # compute the spatial gradient
        #     # y_img_hat = y_hat_off.view(shape).permute(0,3,1,2) # BxC -> NxPxPx2 -> Nx2xHxW
        #     grads = self.sp_grad(y_img_hat) # NxCx2xHxW

        #     gdiff_u_x = F.mse_loss(du_xy[:,0].reshape(N,P,P) / P, grads[:,0,1,...])
        #     gdiff_u_y = F.mse_loss(du_xy[:,1].reshape(N,P,P) / P, grads[:,0,0,...])
        #     gdiff_v_x = F.mse_loss(dv_xy[:,0].reshape(N,P,P) / P, grads[:,1,1,...])
        #     gdiff_v_y = F.mse_loss(dv_xy[:,1].reshape(N,P,P) / P, grads[:,1,0,...])

        #     loss_grads = (gdiff_u_x + gdiff_u_y + gdiff_v_x + gdiff_v_y) / 4

        # # Soft DivFree regularization at high resolution
        # loss_sdiv = 0
        # if self.lam_sdiv > 0:
        #     div_autograd = du_xy[:,0] + dv_xy[:,1]
        #     loss_sdiv = torch.mean(div_autograd**2)

        # if self.lam_curl + self.lam_pde > 0:
        #     w = dv_xy[:,0] - du_xy[:,1]
        #     dw_xy = my_autograd(w, X_hr) # Bx2
            
        # loss_curl = 0
        # if self.lam_curl > 0:
        #     loss_curl = torch.mean(torch.norm(dw_xy, dim=1)**2)

        # loss_pde = 0
        # if self.lam_pde > 0:
        #     left = w*(y_hat_hr[:,0] + y_hat_hr[:,1])
        #     dw_dxy = self.sp_grad(w.reshape(P,P)[None,None,...]) # BxCx2xHxW
        #     d2w_dxy = self.sp_grad(dw_dxy[0,...]) # BxCx2xHxW
        #     lapl_w = d2w_dxy[:,0,1,:,:] + d2w_dxy[:,1,0,:,:]
        #     right = (-(1./3000.)*lapl_w).flatten()
        #     loss_pde = torch.mean(torch.abs(left - right)**2)

        #     if self.current_epoch % 10 == 0:
        #         plt.figure(figsize=(15,5))
        #         plt.suptitle(f'Loss: {loss_pde.item()}')
        #         plt.subplot(131)
        #         plt.imshow(left.reshape(self.H, self.H).detach().cpu())
        #         plt.colorbar()
        #         plt.subplot(132)
        #         plt.imshow(right.reshape(self.H, self.H).detach().cpu())
        #         plt.colorbar()
        #         plt.subplot(133)
        #         plt.imshow(torch.abs(left-right).reshape(self.H, self.H).detach().cpu())
        #         plt.colorbar()
        #         plt.show()

        #     del right
        #     del left

        # # OFF GRID REGULARIZATIAN on SPARSE GRID (MONTECARLO Sampling)
        # # L2 regularization on the gradient of the potential
        # if self.lam_sfn > 0:
        #     # X_random = montecarlo_sampling_xcenters_xincerments(
        #     #     self.n_centers, self.n_increments, self.patch_dim, self.min_l, self.device)
        #     # self.reference_input_hr = patch_sq.view(-1,2)
        #     # shape = X_random.shape # I x C x P x D
        #     # X_random.requires_grad_(True)
        #     # self.H = 256
        #     # patch_ln = torch.linspace(0, 1, self.H)
        #     # patch_sq = torch.meshgrid(patch_ln, patch_ln, device=X_batch.device) # H x H
        #     patch_sq = self.patch_sq_hr.to(X_batch.device) # H x H x 2
        #     size = (self.n_increments, *patch_sq.shape) 
        #     X_incr = patch_sq.unsqueeze(0).expand(size) # I x H x H x D
        #     increments = self.min_l * torch.arange(0,self.n_increments,device=X_batch.device) 
        #     X_incr = X_incr + increments[:,None,None,None] 
            
        #     shape_incr = X_incr.shape
        #     y_incr = self.forward(X_incr.view(-1,2))[0]
        
        # # Sfun regularization
        # loss_sfn = 0
        # if self.lam_sfn > 0:
            
            # y_incr_hat = y_incr.reshape(shape_incr)
            # # compute structure function
            # Sfun2 = torch.mean((y_incr_hat - y_incr_hat[0,...])**2, dim=[3,1,2])
            # loss_sfn = F.mse_loss(  torch.log(Sfun2+self.sfun_eps),
            #                         torch.log(self.sfun_model.to(X_batch.device)+self.sfun_eps))


            # if stage == 'val':
            #     plt.plot(Sfun2.detach().cpu() + self.sfun_eps)
            #     plt.plot(self.sfun_model.detach().cpu()+self.sfun_eps)
            #     plt.show()
            
            # del y_incr
            # del X_incr
            # del patch_sq

        loss = loss_rec + self.lam_pde * loss_pde + self.lam_sdiv * loss_sdiv + self.lam_grads * loss_grads + self.lam_vort_ic * loss_vort_ic
        
        # LOGs, PRINTs and PLOTs
        self.log(f'{stage}/loss/tot',  loss,       on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/rec',  loss_rec,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/pde',  loss_pde,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/sdiv', loss_sdiv,  on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/loss_vort_ic', loss_vort_ic,  on_epoch=True, on_step=False)
        # self.log(f'{stage}/loss/sfn',  loss_sfn,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/grads', loss_grads, on_epoch=True, on_step=False)
        # self.log(f'{stage}/loss/spec', loss_spec,  on_epoch=True, on_step=False)
        # self.log(f'{stage}/loss/curl', loss_curl,  on_epoch=True, on_step=False)


        # raise ValueError('Check log err spectrum')
        # self.log(f'{stage}/metrics/reconstruction', err_rec)
        # self.log(f'{stage}/metrics/angular_degree', err_ang)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.lam_weight)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler,
                "monitor": "train/loss/tot"}