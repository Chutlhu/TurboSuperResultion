import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

import pytorch_lightning as pl

from turboflow.models.core import *
from turboflow.utils import torch_utils as tch
from turboflow.utils import file_utils as fle

from kolmopy.utils import viz_utils as viz
from kolmopy.utils import phy_utils as phy
import turboflow.evaluation as evl

from torchvision import transforms as T
from PIL import Image

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



class plBACON(pl.LightningModule):

    def __init__(self,  name:str,
                        mode:str,
                        mlp_layers_num:int,
                        mlp_layers_dim:int, 
                        do_divfree:bool,
                        pde_diff:float,
                        lam_pde:float, 
                        lam_wrt:float,
                        lam_sdiv:float, 
                        lam_weight:float):

        super().__init__()

        self.save_hyperparameters()

        self.name = name
        self.automatic_optimization = True

        self.mode = mode
        self.do_divfree = do_divfree

        if self.do_divfree:
            nvarout = 1
        else:
            nvarout = 2

        filter_fun = 'Fourier'
        self.mfn = ResMFN(in_dim=3, 
                          hidden_dim=mlp_layers_dim,
                          out_dim=nvarout, 
                          k=mlp_layers_num,
                          data_max_freqs=512)

        # PINN losses
        self.pde_diff = pde_diff
        self.lam_pde = lam_pde
        self.lam_sdiv = lam_sdiv
        self.lam_wrt = lam_wrt
        self.lam_weight = lam_weight
 

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("model")
        group.add_argument("--name", type=str, default="DivFreeRFFNet")        
        group.add_argument("--mode", type=str, default="train")
        group.add_argument("--mlp_layers_num", type=int, default=3)
        group.add_argument("--mlp_layers_dim", type=int, default=256)

        group.add_argument("--do_divfree", type=fle.str2bool, nargs='?', const=True, default=False)

        group.add_argument("--pde_diff", type=float, default=0.)
        group.add_argument("--lam_pde", type=float, default=0)      # 
        group.add_argument("--lam_wrt", type=float, default=0.)
        group.add_argument("--lam_sdiv", type=float, default=0)     # Soft constrain at full resolution
        group.add_argument("--lam_weight", type=float, default=0)   # L2 regularization on the NN's weights (directly in the torch optimizer)
        
        return parent_parser

    def _div_free(self, P, xin):
        P_txy = tch.diff(P, xin)
        u =  P_txy[:, -1] 
        v = -P_txy[:, -2]
        return torch.cat([u[:,None], v[:,None]], dim=-1)
        

    def forward(self, xin, out=0): # x := BxC(Batch, InputChannels)
        xin.requires_grad_(True)

        B = xin.shape[0]

        if self.do_divfree:
            xin.requires_grad_(True)
            P, P_mid = self.mfn(xin)

            u  = self._div_free(P, xin)
            u1 = self._div_free(P_mid[0], xin)
            u2 = self._div_free(P_mid[1], xin)
            u3 = self._div_free(P_mid[2], xin)
            us = (u1, u2, u3)
            return u, us

        return self.mfn(xin, out=out)

    def compute_vorticity(self, y, X):
        u, v = torch.split(y, 1, -1)
    
        du = tch.diff(u, X)
        u_t, u_x, u_y = du.split(1, -1)

        dv = tch.diff(v, X)
        v_t, v_x, v_y = dv.split(1, -1)

        return v_x - u_y


    def set_grad_layer(self, id_layer):
        
        tch.set_grad(self.mfn, False)

        if id_layer == 1:
            tch.set_grad(self.mfn.g0, True)
            tch.set_grad(self.mfn.g1, True)
            tch.set_grad(self.mfn.l1, True)
            tch.set_grad(self.mfn.y1, True)

        elif id_layer == 2:
            tch.set_grad(self.mfn.g2, True)
            tch.set_grad(self.mfn.l2, True)
            tch.set_grad(self.mfn.y2, True)
        
        elif id_layer == 3:
            tch.set_grad(self.mfn.g3, True)
            tch.set_grad(self.mfn.l3, True)
            tch.set_grad(self.mfn.y3, True)

        elif id_layer == 4:
            tch.set_grad(self.mfn.g4, True)
            tch.set_grad(self.mfn.l4, True)
            tch.set_grad(self.mfn.y4, True)

        elif id_layer == 5:
            tch.set_grad(self.mfn.g5, True)
            tch.set_grad(self.mfn.l5, True)
            tch.set_grad(self.mfn.y5, True)


    def _common_step(self, batch, batch_idx:int, stage:str):
        # It is independent of forward
        X_batch, y_batch, y_mid_batch = batch

        B = X_batch.shape[0]
        # H = self.H

        ## FORWARD BATCH
        y_dwn8, y_dwn4, y_dwn2, y_dwn1, y_up2 = self.forward(X_batch, out=0)
        
        ## REGULARIZATION and LOSSES
        # Reconstruction loss
        self.lam_rec = 1

        loss_res_dwn8 = F.mse_loss(y_dwn8, y_mid_batch[0][:,:2])
        loss_res_dwn4 = F.mse_loss(y_dwn4, y_mid_batch[1][:,:2])
        loss_res_dwn2 = F.mse_loss(y_dwn2, y_mid_batch[2][:,:2])
        loss_res_dwn1 = F.mse_loss(y_dwn1, y_batch[:,:2])

        w_dwn8 = self.compute_vorticity(y_dwn8, X_batch)
        w_dwn4 = self.compute_vorticity(y_dwn4, X_batch)
        w_dwn2 = self.compute_vorticity(y_dwn2, X_batch)
        w_dwn1 = self.compute_vorticity(y_dwn1, X_batch)

        loss_wrt_dwn8 = F.mse_loss(w_dwn8, y_mid_batch[0][:,2:])
        loss_wrt_dwn4 = F.mse_loss(w_dwn4, y_mid_batch[1][:,2:])
        loss_wrt_dwn2 = F.mse_loss(w_dwn2, y_mid_batch[2][:,2:])
        loss_wrt_dwn1 = F.mse_loss(w_dwn1, y_batch[:,2:])

        if self.mode == 'pre_train':
            if self.current_epoch < self.its[0]:
                if batch_idx == 0 and self.current_epoch==0: print('Trianing 1st layer - epoch', 0)
                self.set_grad_layer(1)
                y_hat = y_dwn8
                loss_rec = loss_res_dwn8
                loss_wrt = loss_wrt_dwn8

            elif self.current_epoch < self.its[1]:
                if batch_idx == 0 and self.current_epoch==self.its[0]: print('Training 2nd layer - epoch', self.its[0])
                self.set_grad_layer(2)
                y_hat = y_dwn4
                loss_rec = loss_res_dwn4
                loss_wrt = loss_wrt_dwn4

            elif self.current_epoch < self.its[2]:
                if batch_idx == 0 and self.current_epoch == self.its[1]: print('Training 3rd layer - epoch', self.its[1])
                self.set_grad_layer(3)
                y_hat = y_dwn2
                loss_rec = loss_res_dwn2
                loss_wrt = loss_wrt_dwn2

            elif self.current_epoch < self.its[3]:
                if batch_idx == 1 and self.current_epoch == self.its[2]: print('Training 4th layer - epoch', self.its[2])
                self.set_grad_layer(4)
                y_hat = y_dwn1
                loss_rec = loss_res_dwn1
                loss_wrt = loss_wrt_dwn1

        elif self.mode == 'fine_tune':
            tch.set_grad(self.mfn, True)
            tch.set_grad(self.mfn.g5, False)
            tch.set_grad(self.mfn.l5, False)
            tch.set_grad(self.mfn.y5, False)
            y_hat = y_dwn1
            loss_rec = loss_res_dwn1
            loss_wrt = loss_wrt_dwn1

        elif self.mode == 'pde_tune':
            tch.set_grad(self.mfn, True)
            loss_rec = loss_res_dwn1


        # ## off-grid PDE regularization
        # D = 8
        # Rx = 512
        # Rt = 15
        # idx_x = torch.randint(Rx, [D], device=X_batch.device)
        # idx_y = torch.randint(Rx, [D], device=X_batch.device)
        # idx_t = torch.randint(Rt, [D], device=X_batch.device)
        # t = torch.linspace( 0., 1., Rt, device=X_batch.device)[idx_t]
        # x = torch.linspace(-1., 1., Rx, device=X_batch.device)[idx_x]
        # y = torch.linspace(-1., 1., Rx, device=X_batch.device)[idx_y]
        # txy = torch.stack(torch.meshgrid([t, x, y]), dim=-1).view(-1,3)
        # txy.requires_grad_(True)

        # if self.mode == 'pre_train':
        #     if self.current_epoch < self.its[0]:
        #         y_hat = self(txy, out=0)
        #         Rx = (512/4)/8
        #     elif self.current_epoch < self.its[1]:
        #         y_hat = self(txy, out=1)
        #         Rx = (512/4)/4
        #     elif self.current_epoch < self.its[2]:
        #         y_hat = self(txy, out=2)
        #         Rx = (512/4)/2
        #     elif self.current_epoch < self.its[3]:
        #         y_hat = self(txy, out=3)
        #         Rx = (512/4)/1
        #     else:
        #         raise ValueError('Not a good E')

        # elif self.mode == 'fine_tune':
        #     y_hat = self(txy, out=3)
        #     Rx = (512/4)*2

        # elif self.mode == 'pde_tune':
        #     y_hat = self(txy, out=4)
        #     Rx = (512/4)*2

        # else:
        #     raise ValueError('Not a good mode')

        # ## OFF-GRID PDE
        loss_pde = 0
        loss_sdiv = 0

        # if self.lam_sdiv > 0:

        #     u, v = torch.split(y_hat, 1, -1)

        #     du = tch.diff(u, txy)
        #     u_t, u_x, u_y = du.split(1, -1)

        #     dv = tch.diff(v, txy)
        #     v_t, v_x, v_y = dv.split(1, -1)

        #     # Soft Divergence Free equation
        #     res_sdiv = torch.mean(torch.abs(u_x + v_y)**2)
        #     loss_sdiv = res_sdiv

        # if self.lam_pde > 0:
        #     # compute vorticity
        #     w = v_x - u_y

        #     dw = tch.diff(w, txy)
        #     w_t, w_x, w_y = dw.split(1, -1)
        #     ddw = tch.diff(dw, txy)
        #     w_tt, w_xx, w_yy = ddw.split(1, -1)

        #     # spatial change of vorticity / unit volume (u dot Nabla)w
        #     w_spatial = u*w_x + v*w_y
        #     # diffusion of vorticity / univ volume (nu Laplacian w)
        #     diffusion = self.pde_diff
        #     w_diffusion = diffusion*(w_xx + w_yy)

        #     res_pde = w_t + w_spatial - w_diffusion
        #     # loss_pde = torch.mean(torch.abs(res_pde)**2)

        #     # full NS equation
        #     # loss_pde = F.mse_loss(res_pde, torch.zeros_like(res_pde))

        #     res_pde = res_pde.reshape(D,D,D)
        #     res_pde = torch.mean(res_pde**2, dim=[1,2])/(Rx**2)
        #     # causal traning
        #     idx = torch.argsort(t) # sort according to time
        #     res_sorted = res_pde[idx]
    
        #     eps = 10
        #     B = res_pde.shape[0]
        #     M = torch.triu(torch.ones(B, B), diagonal=1).T.to(X_batch.device)
        #     mat = torch.matmul(M, res_sorted)
        #     W = torch.exp(- eps * mat)
        #     loss_pde = torch.mean(W*res_sorted)

        #     if batch_idx == 0 and self.current_epoch % 20 == 0:
        #         plt.plot(tch.to_np(W))
        #         plt.plot(tch.to_np(res_sorted/torch.max(res_sorted)), alpha=0.5)
        #         plt.show()

        # # off-grid spectrum regularization
        # if self.mode == 'pre_train':
        #     if self.current_epoch < self.its[0]:
        #         Rx = (512/4)/8
        #         Rt = 15
        #         dt = 4

        #     elif self.current_epoch < self.its[1]:
        #         Rx = (512/4)/4
        #         Rt = 15
        #         dt = 4

        #     elif self.current_epoch < self.its[2]:
        #         Rx = (512/4)/2
        #         Rt = 15
        #         dt = 4

        #     elif self.current_epoch < self.its[3]:
        #         Rx = (512/4)/1
        #         Rt = 15
        #         dt = 4

        # elif self.mode == 'fine_tune':
        #     Rx = (512/4)/1
        #     Rt = 15
        #     dt = 4

        # elif self.mode == 'pde_tune':
        #     Rx = (512/4)*2
        #     Rt = 15
        #     dt = 1

        # t = torch.linspace( 0., 1., int(Rt), device=X_batch.device)[::dt]
        # x = torch.linspace(-1., 1., int(Rx), device=X_batch.device)
        # y = torch.linspace(-1., 1., int(Rx), device=X_batch.device)
        # txy = torch.stack(torch.meshgrid([t, x, y]), dim=-1)
        # txy.requires_grad_(True)

        # Nt, Nx, Ny, D = txy.shape
        # y_pred = []

        # for t in range(Nt):
        #     xy = txy[t,:,:,:].view(-1,3)
        #     y_pred.append(self.forward(xy).reshape(1,Nx,Ny,2))
            
        # y_pred = torch.cat(y_pred, dim=0)
        
        # S_, k = phy.energy_spectrum(y_pred)
        # S_ = S_
        # S_ = torch.clip(S_, 1e-7, 1)
        # Ek = self.Ek_tgt[:len(k)].to(X_batch.device)
        # Ek = Ek
        # Ek = torch.clip(Ek, 1e-7, 1)

        # loss_spec = F.mse_loss(torch.log(S_), torch.log(Ek))
        loss_spec = 0

        # loss = self.lam_rec * loss_rec + self.lam_pde * loss_pde + self.lam_sdiv * loss_sdiv + self.lam_curl * loss_curl
        self.lam_spec = 1e-2
        loss = loss_rec + self.lam_sdiv * loss_sdiv + self.lam_pde * loss_pde + self.lam_wrt * loss_wrt + self.lam_spec * loss_spec
        
        # LOGs, PRINTs and PLOTs
        self.log(f'{stage}/loss/tot',  loss,       on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/rec',  loss_rec,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/res_s1',  loss_res_dwn1,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/res_s2',  loss_res_dwn2,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/res_s4',  loss_res_dwn4,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/res_s8',  loss_res_dwn8,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/pde',  loss_pde,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/spec', loss_spec,   on_epoch=True, on_step=False)
        self.log(f'{stage}/loss/sdiv', loss_sdiv,  on_epoch=True, on_step=False)

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
        loss_rec = F.mse_loss(y_hat, y_batch)
        return loss_rec

    #  add optional logic at the end of the training
    #  the function is called after every epoch is completed
    def training_epoch_end(self, outputs):
        return None

    def prediction_image_adder(self, x, u, du, w, T, R):
        x = x.reshape(T,R,R,3)
        u = u.reshape(T,R,R,2)

        fig, axarr = plt.subplots(2,T, figsize=(9,6))
        for t in range(T):
            time = x[t,0,0,0]
            axarr[0,t].imshow(tch.to_np(u[t,:,:,0]))
            axarr[0,t].set_title(f'U(t={time})')
            axarr[0,t].axis('off')
            axarr[1,t].imshow(tch.to_np(u[t,:,:,1]))
            axarr[1,t].set_title(f'V(t={time})')
            axarr[1,t].axis('off')
        plt.tight_layout()
        self.logger.experiment.add_figure(f"output/fields", fig, self.current_epoch)
        
    def custom_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def validation_epoch_end(self, outputs):
        ## Save Model graph
        # dummy_input = torch.rand((3,3)).to(self.device)
        # dummy_input.requires_grad_(True)
        # self.logger.experiment.add_graph(self, dummy_input)

        # # LOG IMAGES
        T = 1
        R = 512
        # t = torch.Tensor([0.28571429]).to(self.device)
        t = torch.Tensor([0.]).to(self.device)
        x = torch.linspace(-1, 1, R, device=self.device)
        y = torch.linspace(-1, 1, R, device=self.device)
        txy = torch.stack(torch.meshgrid([t, x, y]), dim=-1).view(-1,3)
        uv_dwn8, uv_dwn4, uv_dwn2, uv_dwn1, uv_up2 = self.forward(txy)

        uv_list = [uv_dwn8, uv_dwn4, uv_dwn2, uv_dwn1, uv_up2]
        scales = ['x0.125', 'x0.25', 'x0.5', 'x1', 'x2']
        res = [(256/2)/8, (256/2)/4, (256/2)/2, (256/2)/1, (256/2)*2]

        for u, uv_ in enumerate(uv_list):

            U = uv_.reshape(T,R,R,2)

            fig = plt.figure(figsize=(10,5))
            Ek, k = phy.energy_spectrum(U)
            Ek, k = tch.to_np(Ek), tch.to_np(k)
            plt.loglog(k, self.Ek_tgt, label='GT')
            plt.loglog(k, Ek, label='Pred')
            plt.axvline(x=res[u], label='Res')
            plt.ylim([1e-15,1])
            plt.tight_layout()
            self.logger.experiment.add_figure(f"output/spec_{scales[u]}", fig, self.current_epoch)

            U = tch.to_np(U)
            fig, axarr = plt.subplots(1,2,figsize=(12,8))
            axarr[0].imshow(U[0,:,:,0])
            axarr[0].set_title(f'U(t={t})')
            axarr[0].axis('off')
            axarr[1].imshow(U[0,:,:,1])
            axarr[1].set_title(f'V(t={t})')
            axarr[1].axis('off')
            plt.tight_layout()
            self.logger.experiment.add_figure(f"output/uv_{scales[u]}", fig, self.current_epoch)

        # T = 2
        # R = 32
        # t = torch.Tensor([0.5, 0.6]).to(self.device)
        # x = torch.linspace(-0., 0.125, R, device=self.device)
        # y = torch.linspace(-0., 0.125, R, device=self.device)
        # txy = torch.stack(torch.meshgrid([t, x, y]), dim=-1).view(-1,3)

        # txy.requires_grad_(True)
        # uv_ = self(txy)

        # u, v = torch.split(uv_, 1, -1)
        # du = tch.diff(u, txy)
        # u_t, u_x, u_y = du.split(1, -1)

        # dv = tch.diff(v, txy)
        # v_t, v_x, v_y = dv.split(1, -1)

        # w = v_x - u_y

        # dw = tch.diff(w, txy)
        # w_t, w_x, w_y = dw.split(1, -1)
        # ddw = tch.diff(dw, txy)
        # w_tt, w_xx, w_yy = ddw.split(1, -1)

        # # spatial change of vorticity / unit volume (u dot Nabla)w (advection)
        # w_a = u*w_x + v*w_y
        # # diffusion of vorticity / univ volume (nu Laplacian w)
        # diffusion = self.pde_diff
        # w_d = diffusion*(w_xx + w_yy)

        # r_w = torch.abs(w_t + w_a - w_d)**2
        # r_d = torch.abs(u_x + v_y)**2

        # W = tch.to_np(w.reshape(T,R,R))
        # W_t = tch.to_np(w_t.reshape(T,R,R))
        # W_s = tch.to_np(w_a.reshape(T,R,R))
        # W_d = tch.to_np(w_d.reshape(T,R,R))
        # D = tch.to_np(r_d.reshape(T,R,R))
        # W_r = tch.to_np(r_w.reshape(T,R,R))

        # h = 5
        # figsize = (T*h, h)
        
        # fig, axarr = plt.subplots(1, T, figsize=figsize)
        # for t in range(T):
        #     im = axarr[t].imshow(D[t,...])
        #     plt.colorbar(im, ax=axarr[t])
        #     axarr[t].axis('off')
        # plt.tight_layout()
        # self.logger.experiment.add_figure(f"residuals/sdiv", fig, self.current_epoch)

        # fig, axarr = plt.subplots(1, T, figsize=figsize)
        # for t in range(T):
        #     im = axarr[t].imshow(W[t,...])
        #     plt.colorbar(im, ax=axarr[t])
        #     axarr[t].axis('off')
        # plt.tight_layout()
        # self.logger.experiment.add_figure(f"output/w", fig, self.current_epoch)

        # fig, axarr = plt.subplots(4, T, figsize=(6,12))
        # for t in range(T):
        #     im = axarr[0,t].imshow(W_t[t,...])
        #     plt.colorbar(im, ax=axarr[0,t])
        #     axarr[0,t].axis('off')
            
        #     im = axarr[1,t].imshow(W_s[t,...])
        #     plt.colorbar(im, ax=axarr[1,t])
        #     axarr[1,t].axis('off')

        #     im = axarr[2,t].imshow(W_d[t,...])
        #     plt.colorbar(im, ax=axarr[2,t])
        #     axarr[2,t].axis('off')
            
        #     im = axarr[3,t].imshow(W_r[t,...])
        #     plt.colorbar(im, ax=axarr[3,t])
        #     axarr[3,t].axis('off')
        # plt.tight_layout()
        # self.logger.experiment.add_figure(f"residuals/pde", fig, self.current_epoch)

        
        # self.custom_histogram_adder()
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.lam_weight)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.milestones, gamma=0.1)
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler,
                "monitor": "train/loss/tot"}


    def _configure_optim_backbone(self):
        # return optimizers and schedulers for pre-training
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler,
                "monitor": "train/loss/res_s1"}

    def _configure_optim_finetune(self):
        # return optimizers and scheduler for fine-tine
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.lam_weight)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler,
                "monitor": "train/loss/res_s1"}

    def _configure_optim_pde(self):
        # return optimizers and scheduler for fine-tine
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.lam_weight)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True)
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler,
                "monitor": "train/loss/res_s1"}

    def configure_optimizers(self):
        if self.mode == 'pre_train':
            print('PRE TRAIN')
            return self._configure_optim_backbone()
        elif self.mode == 'fine_tune':
            print('FINE TUNING')
            return self._configure_optim_finetune()
        elif self.mode == 'pde_tune':
            print('PDE TUNING')
            return self._configure_optim_pde()

# class plDivFreeRFFNet(pl.LightningModule):

#     def __init__(self,  name:str,
#                         do_time:bool,
#                         do_cnn:bool,
#                         mode:str,
#                         mlp_layers_num:int, mlp_layers_dim:int, 
#                         mlp_last_actfn:str,
#                         rff_num_space:int, rff_scale_space:float,
#                         rff_num_time:int, rff_scale_time:float,
#                         do_divfree:bool,
#                         lam_pde:float, 
#                         lam_sdiv:float,
#                         lam_sfn:float,
#                         lam_spec:float,
#                         lam_curl:float,
#                         lam_grads:float, lam_weight:float,
#                         sfn_min_x:float, sfn_num_centers:int, sfn_num_increments:int, sfn_patch_dim:int):

#         super(plDivFreeRFFNet, self).__init__()

#         self.save_hyperparameters()

#         self.name = name
#         self.automatic_optimization = True

#         if do_time:
#             nvarin = 3
#         else:
#             nvarin = 2

#         self.do_rff = True
#         self.do_cnn = do_cnn
#         self.do_divfree = do_divfree

#         # # space subnet
#         # self.rff_time = Fourier(rff_num_time, rff_scale_time, nvars=1)
#         # self.rff_space = Fourier(rff_num_space, rff_scale_space, nvars=2)
        
#         if self.do_divfree:
#             self.div = DivFree()
#             nvarout = 1
#         else:
#             nvarout = 2

    
#         filter_fun = 'Fourier'
#         self.mfn = ResMFN(in_dim=3, hidden_dim=mlp_layers_dim, out_dim=2, k=mlp_layers_num, 
#                             filter_fun=filter_fun, data_max_freqs=512)

#         # off-grid regularization params
#         self.min_l = sfn_min_x
#         self.n_centers = sfn_num_centers
#         self.patch_dim = sfn_patch_dim
#         self.n_increments = sfn_num_increments
#         self.sfun = lambda x, y : torch.mean(torch.abs(x - y).pow(2))
#         self.sfun_model = 0.25*0.00275*torch.arange(self.n_increments)**2
#         self.sfun_eps = 1e-20
#         # self.spec_model = torch.load(root_dir + 'data/hr_spect256.pt')
        
#         # self.sp_grad = SpatialGradient(mode='diff', order=1, normalized=True)
#         # self.sp_lapl = Laplacian(kernel_size=5, normalized=True)

#         self.Re = 3000
        
#         # PINN losses
#         self.lam_pde = lam_pde
#         self.lam_sdiv = lam_sdiv
#         self.lam_sfn = lam_sfn
#         self.lam_spec = lam_spec
#         self.lam_grads = lam_grads
#         self.lam_curl = lam_curl
#         self.lam_weight = lam_weight
#         self.lam_vort_ic = 0
        
#         # # reference image for logging on tensorboard
#         # self.L = 64
#         # patch_ln_lr = torch.linspace(0, 1, self.L)
#         # patch_sq_lr = torch.stack(torch.meshgrid(patch_ln_lr, patch_ln_lr), dim=-1)
#         # self.reference_input_lr = patch_sq_lr.view(-1,2)
#         # self.H = 256
#         # patch_ln_hr = torch.linspace(0, 1, self.H)
#         # patch_sq_hr = torch.stack(torch.meshgrid(patch_ln_hr, patch_ln_hr), dim=-1)
#         # self.patch_sq_hr = patch_sq_hr
#         # self.reference_input_hr = patch_sq_hr.view(-1,2)

#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         group = parent_parser.add_argument_group("model")
#         group.add_argument("--name", type=str, default="DivFreeRFFNet")
        
#         group.add_argument("--do_time",     type=fle.str2bool, nargs='?', const=True, default=False)
        
#         group.add_argument("--mode", type=str, default="train")
        
#         group.add_argument("--do_cnn",     type=fle.str2bool, nargs='?', const=True, default=False)
#         group.add_argument("--mlp_layers_num", type=int, default=3)
#         group.add_argument("--mlp_layers_dim", type=int, default=256)
#         group.add_argument("--mlp_last_actfn", type=str, default="tanh")

#         group.add_argument("--do_divfree", type=fle.str2bool, nargs='?', const=True, default=False)
#         group.add_argument("--rff_num", type=int, default=256)
#         group.add_argument("--rff_scale", type=float, default=10)
        
#         group.add_argument("--lam_pde", type=float, default=0)      # 
#         group.add_argument("--lam_sdiv", type=float, default=0)     # Soft constrain at full resolution
#         group.add_argument("--lam_curl", type=float, default=0)     # || grad w ||_2^2
#         group.add_argument("--lam_spec", type=float, default=0)     # Target spectrogram at full resolution
#         group.add_argument("--lam_grads", type=float, default=0)    # Autograd = Spatial Gradient at Full-Resolution
#         group.add_argument("--lam_weight", type=float, default=0)   # L2 regularization on the NN's weights (directly in the torch optimizer)
        
#         group.add_argument("--lam_sfn", type=float, default=0)
#         group.add_argument("--sfn_min_x", type=float, default=1./256.)
#         group.add_argument("--sfn_num_centers", type=int, default=5)
#         group.add_argument("--sfn_patch_dim", type=int, default=32)
#         group.add_argument("--sfn_num_increments", type=int, default=5)
#         return parent_parser

#     def forward(self, xin): # x := BxC(Batch, InputChannels)
#         xin.requires_grad_(True)

#         B = xin.shape[0]

#         if self.do_divfree:
#             xin.requires_grad_(True)
#             P, _ = self.mfn(xin)
#             P_txy = tch.diff(P, xin)

#             u =  P_txy[:, 2]
#             v = -P_txy[:, 1]
#             u_df = torch.cat([u[:,None], v[:,None]], dim=-1)

#             du = tch.diff(u, xin)
#             u_t, u_x, u_y = du.split(1, -1)
#             dv = tch.diff(v, xin)
#             v_t, v_x, v_y = dv.split(1, -1)
#             w = v_x - u_y

#             return u_df, w, P_txy

#         u, us = self.mfn(xin)

#         # u1, v1 = u.split(1, -1)
#         # du = tch.diff(u1, xin)
#         # u_t, u_x, u_y = du.split(1, -1)
#         # dv = tch.diff(v1, xin)
#         # v_t, v_x, v_y = dv.split(1, -1)
#         # w = v_x - u_y

#         return u, us


#     def _common_step(self, batch, batch_idx:int, stage:str):
#         # It is independent of forward
#         X_batch, y_batch, y_mid_batch = batch

#         # B = X_batch.shape[0]
#         # H = self.H

#         ## FORWARD BATCH
#         y_hat, y_mid_hat = self.forward(X_batch)

#         ## REGULARIZATION and LOSSES
#         # Reconstruction loss
#         self.lam_rec = 1
        
#         assert y_hat.shape == y_batch[:,:2].shape
#         # loss_rec = F.mse_loss(y_hat, y_batch[:,:2])

#         loss_res_s8 = F.mse_loss(y_mid_hat[0], y_mid_batch[0])
#         loss_res_s4 = F.mse_loss(y_mid_hat[1], y_mid_batch[1])
#         loss_res_s2 = F.mse_loss(y_mid_hat[2], y_mid_batch[2])
#         loss_res_s1 = F.mse_loss(y_hat, y_batch)

#         def to_np(x) : return x.detach().cpu().numpy()

#         its = [200, 400, 800, 1200]
#         its = [50, 100, 150, 200]

#         # B = X_batch.shape[0]
#         # res_rec = torch.mean(torch.abs(y_mid_hat[0] - y_mid_batch[0])**2, dim=1)    
#         # idx = torch.argsort(X_batch[:,0]) # sort according to time
#         # res_sorted = res_rec[idx]
#         # tol = 10
#         # times = X_batch[:,0]
#         # val, count = torch.unique(times, return_counts=True)
#         # c = count[0]
#         # sub_m = [torch.zeros(c, B, device=X_batch.device)]
#         # w = 0
#         # for i in range(1,len(count)):
#         #     h = count[i]
#         #     w += count[i-1]
#         #     m = torch.hstack([torch.ones(h, w, device=X_batch.device), 
#         #                         torch.zeros(h, B-w, device=X_batch.device)])
#         #     sub_m.append(m)
#         # M = torch.vstack(sub_m)
#         # W = torch.exp(- tol * (torch.matmul(M, res_sorted))).detach()
#         # res_rec = W*res_sorted
#         # if batch_idx == 0 and self.current_epoch % 10 == 0 and stage == 'train':
#         #     plt.semilogy(to_np(W))
#         #     plt.plot(to_np(res_sorted), alpha=0.7)
#         #     plt.plot(to_np(res_rec), alpha=0.7)
#         #     plt.show()
#         # loss_res_s8 = F.mse_loss(res_rec, torch.zeros_like(res_rec))

#         def set_grad_layer(id_layer):
            
#             tch.set_grad(self.mfn, False)

#             if id_layer == 1:
#                 tch.set_grad(self.mfn.g0, True)
#                 tch.set_grad(self.mfn.g1, True)
#                 tch.set_grad(self.mfn.l1, True)
#                 tch.set_grad(self.mfn.y1, True)

#             elif id_layer == 2:
#                 tch.set_grad(self.mfn.g2, True)
#                 tch.set_grad(self.mfn.l2, True)
#                 tch.set_grad(self.mfn.y2, True)
            
#             elif id_layer == 3:
#                 tch.set_grad(self.mfn.g3, True)
#                 tch.set_grad(self.mfn.l3, True)
#                 tch.set_grad(self.mfn.y3, True)

#             elif id_layer == 4:
#                 tch.set_grad(self.mfn.g4, True)
#                 tch.set_grad(self.mfn.l4, True)
#                 tch.set_grad(self.mfn.y4, True)


#         if self.mode == 'pre_train' and self.current_epoch < its[0]:
#             if batch_idx == 0 and self.current_epoch==0: print('here 0')
#             set_grad_layer(1)
#             loss_rec = loss_res_s8

#         elif self.mode == 'pre_train' and self.current_epoch < its[1]:
#             if batch_idx == 0 and self.current_epoch==its[0]: print('here', its[0])
#             set_grad_layer(2)
#             loss_rec = loss_res_s4

#         elif self.mode == 'pre_train' and self.current_epoch < its[2]:
#             if batch_idx == 0 and self.current_epoch == its[1]: print('here', its[1])
#             set_grad_layer(3)
#             loss_rec = loss_res_s2

#         elif self.mode == 'pre_train' and self.current_epoch < its[3]:
#             if batch_idx == 1 and self.current_epoch == its[2]: print('here', its[2])
#             set_grad_layer(4)
#             loss_rec = loss_res_s1

        
#         if self.mode == 'fine_tune':
#             tch.set_grad(self.mfn, True)
#             loss_rec =  loss_res_s1


#         # ## OFF-GRID PDE
#         loss_pde = 0
#         loss_sdiv = 0
#         loss_curl = 0

#         # X_off_hr = torch.rand([B, 3]).to(X_batch.device)
#         # X_off_hr[:,0] = 2*X_off_hr[:,0]     # t \in [0,2]
#         # X_off_hr[:,1] = 2*X_off_hr[:,1] - 1 # x,y \in [-1,1]
#         # X_off_hr[:,2] = 2*X_off_hr[:,2] - 1 # x,y \in [-1,1]
#         # y_off_hr, _ = self.forward(X_off_hr)
#         # W = torch.triu(torch.ones(B,B).to(X_batch.device), diagonal=1).T

#         # if self.lam_sdiv + self.lam_pde > 0:
#         #     X_off_hr = X_batch
#         #     y_off_hr = y_hat

#         #     # Navier Stokes loss (vorticity equation)
#         #     u, v = torch.split(y_off_hr, 1, -1)
            
#         #     du = tch.diff(u, X_off_hr)
#         #     u_t, u_x, u_y = du.split(1, -1)

#         #     dv = tch.diff(v, X_off_hr)
#         #     v_t, v_x, v_y = dv.split(1, -1)

#         #     w = v_x - u_y

#         #     dw = tch.diff(w, X_off_hr)
#         #     w_t, w_x, w_y = dw.split(1, -1)
#         #     ddw = tch.diff(dw, X_off_hr)
#         #     w_tt, w_xx, w_yy = ddw.split(1, -1)

#         #     # spatial change of vorticity / unit volume (u dot Nabla)w
#         #     w_spatial = u*w_x + v*w_y
#         #     # diffusion of vorticity / univ volume (nu Laplacian w)
#         #     diffusion = 0 # 1/self.Re
#         #     w_diffusion = diffusion*(w_xx + w_yy)

#         #     # full NS equation
#         #     res_pde = torch.abs(w_t + w_spatial + w_diffusion)**2

#         #     loss_pde = F.mse_loss(res_pde, torch.zeros_like(res_pde))
            
#         #     res_sdiv = u_x + v_y
#         #     loss_sdiv = F.mse_loss(res_sdiv, torch.zeros_like(res_sdiv))
            

#         # loss = self.lam_rec * loss_rec + self.lam_pde * loss_pde + self.lam_sdiv * loss_sdiv + self.lam_curl * loss_curl
#         loss = loss_rec
        
#         # LOGs, PRINTs and PLOTs
#         self.log(f'{stage}/loss/tot',  loss,       on_epoch=True, on_step=False)
#         self.log(f'{stage}/loss/rec',  loss_rec,   on_epoch=True, on_step=False)
#         self.log(f'{stage}/loss/res_s1',  loss_res_s1,   on_epoch=True, on_step=False)
#         self.log(f'{stage}/loss/res_s2',  loss_res_s2,   on_epoch=True, on_step=False)
#         self.log(f'{stage}/loss/res_s4',  loss_res_s4,   on_epoch=True, on_step=False)
#         self.log(f'{stage}/loss/res_s8',  loss_res_s8,   on_epoch=True, on_step=False)
#         # self.log(f'{stage}/loss/pde',  loss_pde,   on_epoch=True, on_step=False)
#         # self.log(f'{stage}/loss/curl', loss_curl,  on_epoch=True, on_step=False)
#         # self.log(f'{stage}/loss/sdiv', loss_sdiv,  on_epoch=True, on_step=False)

#         return loss

#     def training_step(self, batch, batch_idx):
#         return self._common_step(batch, batch_idx, 'train')
    
#     def validation_step(self, batch, batch_idx):
#         torch.set_grad_enabled(True) # needed for divergenge
#         return self._common_step(batch, batch_idx, 'val')

#     def test_step(self, batch, batch_idx):
#         torch.set_grad_enabled(True) # needed for divergenge
#         X_batch, y_batch = batch
#         y_hat, Py_hat = self.forward(X_batch)

#         err_rec = evl.recostruction_error(y_hat, y_batch, avg=True)
#         err_ang = evl.angular_error_2Dfield(y_hat, y_batch, avg=True)
#         self.log('test/metrics/reconstruction', err_rec)
#         self.log('test/metrics/angular_degree', err_ang)
#         loss_rec = F.mse_loss(y_hat, y_batch)
#         return loss_rec

#     #  add optional logic at the end of the training
#     #  the function is called after every epoch is completed
#     def training_epoch_end(self, outputs):
#         return None

#     def prediction_image_adder(self,x,u,Pu,R,res):
#         x = x.reshape(R,R,2)
#         u = u.reshape(R,R,2)
#         Pu = Pu.reshape(R,R)
#         self.logger.experiment.add_image(f"input/{res}/x", torch.Tensor.cpu(x[:,:,0]),
#                                          self.current_epoch,dataformats="HW")
#         self.logger.experiment.add_image(f"input/{res}/y", torch.Tensor.cpu(x[:,:,1]),
#                                          self.current_epoch,dataformats="HW")
#         self.logger.experiment.add_image(f"output/{res}/ux", torch.Tensor.cpu(u[:,:,0]),
#                                          self.current_epoch,dataformats="HW")
#         self.logger.experiment.add_image(f"output/{res}/uy", torch.Tensor.cpu(u[:,:,1]),
#                                          self.current_epoch,dataformats="HW")
#         self.logger.experiment.add_image(f"output/{res}/Pu", torch.Tensor.cpu(Pu),
#                                          self.current_epoch,dataformats="HW")

#     def custom_histogram_adder(self):
#         # iterating through all parameters
#         for name,params in self.named_parameters():
#             self.logger.experiment.add_histogram(name,params,self.current_epoch)

#     def validation_epoch_end(self, outputs):
#         # ## Save Model graph
#         # dummy_input = torch.rand((3,2))
#         # dummy_input.requires_grad_(True)
#         # self.logger.experiment.add_graph(plDivFreeRFFNet(self.hparams), dummy_input)

#         # LOG IMAGES
#         # self.prediction_image_adder(x_hr, u_hr, Pu_hr, 256, 'hr')
        
#         # self.custom_histogram_adder()
#         pass

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.lam_weight)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.milestones, gamma=0.1)
#         return {"optimizer": optimizer, 
#                 "lr_scheduler": scheduler,
#                 "monitor": "train/loss/tot"}
