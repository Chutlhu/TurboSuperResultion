import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


#####################################
###     Random Fourier Features
##
#

class Fourier(nn.Module):
	def __init__(self, nmb=256, scale=10):

		super(Fourier, self).__init__()
		self.b = torch.randn(2, nmb)*scale
		self.pi = 3.14159265359

	def forward(self, x):
		x = torch.matmul(2*self.pi*x, self.b.to(x.device))
		return torch.cat([torch.sin(x), torch.cos(x)], -1)


class RFFNet_pl(pl.LightningModule):
    def __init__(self, layer_dimension, f_nfeatures, f_scale, device, lam_pde=1e-4):
        super().__init__()
        
        layers = []
        num_layers = len(layer_dimension)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(create_blockReLU(layer_dimension[l], layer_dimension[l+1]))
            
        blocks.append(nn.Linear(layer_dimension[-2], layer_dimension[-1]))
        blocks.append(nn.Tanh())
        
        self.rff = Fourier(f_nfeatures, f_scale)
        self.mlp = nn.Sequential(*blocks)
        
        # # Fourier features
        # self.B = torch.from_numpy(self.B).float().to(device)
        
        # PINN losses
        self.lam_pde = lam_pde
    
    
    def forward(self, x):
        x = self.rff(x)
        x = self.mlp(x)
        # in lightning, forward defines the prediction/inference actions
        return x
    
    
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, x_true = batch
        # x.requires_grad_(True)
        x_pred = self.forward(x)
        
        
        # # physics based loss - div = 0
        # if self.lam_pde > 0:
        #     u, v = torch.split(x_pred,1,-1)
        #     du_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]       
        #     dv_y = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
        #     loss_div = torch.norm(du_x[...,0] + dv_y[...,1])
        # else:
        loss_div = 0
    
        # reconstruction loss
        loss_rec = F.mse_loss(x_pred, x_true)

        # losses 
        loss = loss_rec + self.lam_pde*loss_div
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        self.log('train_loss_data', loss_rec)
        self.log('train_loss_div', loss_div)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, x_true = batch
        x_pred = self.forward(x)
        loss = F.mse_loss(x_pred, x_true)
        psnr = 10 * np.log(2*loss.item())
        self.log('valid_loss', loss, on_step=True)
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


#####################################
###     Fourier Neural Operator
##
#

class SpectralConv2d(nn.Module):
    """
    From https://github.com/zongyi-li/fourier_neural_operator
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, inputs, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the locations (x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):

        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        
        x = self.fc0(x)
        
        x = x.permute(0, 3, 1, 2) # not the output of the fc are stacked as channels # B x C x X x Y

        x1 = self.conv0(x) # conv1
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1) # from B x C x X x Y to B x X x Y x C
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        return x  

    
class FNONet_pl(pl.LightningModule):
    def __init__(self, modes1, modes2,  width, lam_pde):
        super().__init__()
        
        self.network = FNO2d(modes1, modes2,  width)
        self.lam_pde = lam_pde
        
    def forward(self, x):
        return self.network(x) 
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, x_true = batch
        x.requires_grad_(True)
        x_pred = self.forward(x)
        
        # physics based loss - div = 0
        u, v = torch.split(x_pred,1,-1)
        du_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]       
        dv_y = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
        
        # PDE Loss
        loss_div = torch.norm(du_x[...,0] + dv_y[...,1])
    
        # reconstruction loss
        loss_rec = F.mse_loss(x_pred, x_true)

        # losses 
        loss = loss_rec + self.lam_pde*loss_div
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_true = batch
        x_pred = self.forward(x)
        loss = F.mse_loss(x_pred, x_true)
        self.log('valid_loss', loss, on_step=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == '__main__':
    x = torch.randn((1, 256, 256, 2))
    layers = [2, 256, 256, 3]
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    net = FFN(layers, device).to(device)
    pred = net(x.to(device))
    print(pred)
