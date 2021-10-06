import torch
import torch.nn as nn
import torch.nn.functional as F

'''
From
https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/classification_demo.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from turboflow.models.basics import LinearReLU, LinearTanh, Fourier

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


def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi


class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, n_centers, out_features, basis_func):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.n_centers = n_centers
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(n_centers, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(n_centers))
        self.alphas = nn.Parameter(torch.Tensor(n_centers, out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.normal_(self.alphas, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.n_centers, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return torch.tanh(self.basis_func(distances) @ self.alphas)


class DivFreeRBF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, x):
        # compute first order deriv
        dy_xy = torch.autograd.grad(y, x, torch.ones_like(y), 
                                    create_graph=True,
                                    retain_graph=True)[0]
        dy_x, dy_y = dy_xy.split(1,-1)
        # compute secord order deriv
        dy_x_xy = torch.autograd.grad(dy_x, x, torch.ones_like(dy_x), 
                                    create_graph=True,
                                    retain_graph=True)[0]
        dy_xx, dy_xy = dy_x_xy.split(1,-1)
        dy_y_xy = torch.autograd.grad(dy_y, x, torch.ones_like(dy_y),
                                    create_graph=True,
                                    retain_graph=True)[0]
        dy_yx, dy_yy = dy_y_xy.split(1,-1)

        # gather results in a matrix Bx2x2 in the form of Div-free kernel
        # K1 = torch.cat([-dy_yy, dy_xy], dim=-1)[...,None]
        # K2 = torch.cat([dy_yx, -dy_xx], dim=-1)[...,None]
        # K = torch.cat([K1, K2], dim=-1)
        # the columns of K make a divergence-free field
        u =  dy_xy - dy_yy
        v =  dy_xy - dy_xx
        return torch.tanh(torch.cat([u, v], dim=-1))


class RBFNet(nn.Module):
    
    def __init__(self, name, dim_mpl_layers, f_nfeatures, f_scale, n_centers, n_out):

        super(RBFNet, self).__init__()
        self.name = name
        
        # regression/pinn network       
        self.rff = Fourier(f_nfeatures, f_scale) # directly the random matrix 'cause of checkpoint and load
        self.mlp = MLP(dim_mpl_layers)
        basis_func = lambda x : gaussian(x)
        self.rbf = RBF(dim_mpl_layers[-1], n_centers, n_out, basis_func)
        self.divfree = DivFreeRBF()
        
    
    def forward(self, x): # x := BxC(Batch, InputChannels)
        # xin.requires_grad_(True)
        ## implement periodicity
        x = torch.remainder(x,1)
        ## Fourier features
        x = self.rff(x) # Batch x Fourier Features
        ## MLP
        x = self.mlp(x)
        ## RBF
        x = self.rbf(x)
        ## DivFree interpolation
        # x = self.divfree(x, xin)
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
                y_hat1 = self.forward(x_batch)
                loss = F.mse_loss(y_hat1, y_batch)
                current_loss += (1/batches) * (loss.item() - current_loss)
                loss.backward()
                optimiser.step()
                progress += y_batch.size(0)
                if epoch % 100 == 0:
                    print('Epoch: %d, Loss: %f' % (epoch, current_loss))
        print('Done with Training')
        print('Final error:', current_loss)

# class RBFNet_divFree(nn.Module):
#     def __init__(self, num_features=2, num_centers=100, device='cpu'):
#         super(RBFNet_divFree, self).__init__()

#         centers = torch.Tensor(num_centers, num_features)
#         self.centers = nn.Parameter(centers).to(device)
#         sigmas = torch.Tensor(num_centers)
#         self.sigmas = nn.Parameter(sigmas).to(device)
#         weights = torch.Tensor(num_centers, num_features)
#         self.weights = nn.Parameter(weights).to(device)
        
#         # initialize weights and biases
#         nn.init.kaiming_uniform_(self.centers, a=2.236) # centers init
#         nn.init.kaiming_uniform_(self.weights, a=2.236) # weights init
        
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.centers)
#         bound = 1 / fan_in ** 0.5
#         nn.init.uniform_(self.sigmas, 0, bound)  # bias init
        
#         self.D = num_features
#         self.eye = torch.eye(self.D).to(device)

#     def kernel_fun(self, batches):
#         cdist = torch.cdist(batches, self.centers) # N x C
#         invSigma2 = 1/(self.sigmas**2) # 1 x C
#         exp = torch.exp(-0.5*invSigma2*(cdist**2)) # N x C
#         cdiff = batches[:,None,:] - self.centers[None,:,:]
#         arg1 = torch.matmul(cdiff[...,None], cdiff[...,None,:])
#         arg2 = ((self.D - 1) - invSigma2*(cdist**2))[...,None,None] * self.eye[None,None,...]
#         kernel = invSigma2[...,None,None] * exp[...,None,None] * (arg1 + arg2)
#         return torch.einsum('ncdd,cd->nd', kernel, self.weights)

#     def forward(self, batches):
#         return self.kernel_fun(batches)