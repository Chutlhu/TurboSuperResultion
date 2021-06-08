import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Fourier feature mapping
def input_mapping(x, B): 
    if B is None:
        return x
    else:
        x_proj = torch.matmul(2.*np.pi*x, B.T) # B x F
        features =  torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        return features

def input_wavelet_mapping(x, B): 
    if B is None:
        return x
    else:
        x_proj = torch.matmul(2.*np.pi*x, B.T) # B x F
        wavelet_basis = 
        features =  torch.cat([torch.real(x_proj), torch.imag(x_proj)], axis=-1)
        return features

# Simple Network
def create_blockReLU(n_in, n_out):
    # do not work with ModuleList here either.
    block = nn.Sequential(
      nn.Linear(n_in, n_out),
      nn.ReLU()
    )
    return block


class FFN(pl.LightningModule):
    def __init__(self, layer_channels, mapping, device):
        super().__init__()
        
        if mapping is None:
            self.B = None
            input_features = 2
            layer_channels.insert(0, input_features)
        else:
            self.mapping = mapping
            B = mapping['B']
            input_features = 2*B.shape[0]
            layer_channels.insert(0, input_features)
            self.B = torch.from_numpy(B).float().to(device)

        layers = []
        num_layers = len(layer_channels)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(create_blockReLU(layer_channels[l], layer_channels[l+1]))
            
        blocks.append(nn.Linear(layer_channels[-2], layer_channels[-1]))
        blocks.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*blocks)
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = input_mapping(x, self.B)
        return self.mlp(x) 
    
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, x_true = batch
        x_pred = self.forward(x)
        loss = F.mse_loss(x_pred, x_true)
        psnr = 10 * np.log(2*loss.item())
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
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



class FFN_Improved(pl.LightningModule):
    """
    On the eigenvector bias of Fourier feature networks: From regressionto solving multi-scale PDEs with physics-informed neural networks
    S.  Wang,  H.  Wang  and  P.  Perdikaris, 2021 
    https://doi.org/10.1016/j.cma.2021.113938
    """
    def __init__(self, layer_channels, mapping1, mapping2, mapping3, device):
        super().__init__()
    
        self.mapping = mapping1
        B = self.mapping['B']
        input_features = 6*B.shape[0]
        layer_channels.insert(0, input_features)
        self.B1 = torch.from_numpy(mapping1['B']).float().to(device)
        self.B2 = torch.from_numpy(mapping2['B']).float().to(device)
        self.B3 = torch.from_numpy(mapping3['B']).float().to(device)

        layers = []
        num_layers = len(layer_channels)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(create_blockReLU(layer_channels[l], layer_channels[l+1]))
            
        blocks.append(nn.Linear(layer_channels[-2], layer_channels[-1]))
        blocks.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*blocks)
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # one possibility is to have different B with different scales
        x1 = input_mapping(x, self.B1)
        x2 = input_mapping(x, self.B2)
        x3 = input_mapping(x, self.B3)
        x = torch.cat([x1, x2, x3], axis=-1)
        # one possibility is to have different B with different scales
        return self.mlp(x) 
    
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, x_true = batch
        x_pred = self.forward(x)
        loss = F.mse_loss(x_pred, x_true)
        psnr = 10 * np.log(2*loss.item())
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
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

class WFN(pl.LightningModule):
    """
    On the eigenvector bias of Fourier feature networks: From regressionto solving multi-scale PDEs with physics-informed neural networks
    S.  Wang,  H.  Wang  and  P.  Perdikaris, 2021 
    https://doi.org/10.1016/j.cma.2021.113938
    """
    def __init__(self, layer_channels, mapping1, mapping2, mapping3, device):
        super().__init__()
    
        self.mapping = mapping1
        B = self.mapping['B']
        input_features = 6*B.shape[0]
        layer_channels.insert(0, input_features)
        self.B1 = torch.from_numpy(mapping1['B']).float().to(device)
        self.B2 = torch.from_numpy(mapping2['B']).float().to(device)
        self.B3 = torch.from_numpy(mapping3['B']).float().to(device)

        layers = []
        num_layers = len(layer_channels)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(create_blockReLU(layer_channels[l], layer_channels[l+1]))
            
        blocks.append(nn.Linear(layer_channels[-2], layer_channels[-1]))
        blocks.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*blocks)
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # one possibility is to have different B with different scales
        x1 = input_mapping(x, self.B1)
        x2 = input_mapping(x, self.B2)
        x3 = input_mapping(x, self.B3)
        x = torch.cat([x1, x2, x3], axis=-1)
        # one possibility is to have different B with different scales
        return self.mlp(x) 
    
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, x_true = batch
        x_pred = self.forward(x)
        loss = F.mse_loss(x_pred, x_true)
        psnr = 10 * np.log(2*loss.item())
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
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



if __name__ == '__main__':
    x = torch.randn((1, 256, 256, 2))
    layers = [2, 256, 256, 3]
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    net = FFN(layers, device).to(device)
    pred = net(x.to(device))
    print(pred)
