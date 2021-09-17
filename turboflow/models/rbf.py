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


class RBFNet(nn.Module):
    def __init__(self, in_features, out_features, n_centers):
        super(RBFNet, self).__init__()

        self.coeffs = nn.Parameter(torch.Tensor(n_centers, out_features))
        self.centers = nn.Parameter(torch.Tensor(n_centers, in_features)) # C x F
        self.log_sigmas = nn.Parameter(torch.Tensor(n_centers)) # C
        self.basis_fun = lambda x : torch.exp(-1*x.pow(2))
        self.in_features = in_features
        self.out_features = out_features
        self.n_centers = n_centers
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.log_sigmas, 0.5)
        nn.init.kaiming_uniform_(self.coeffs, a=2.236) # coefficients init

    def forward(self, x):
        size = (x.size(0), self.n_centers, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        x = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        x = self.basis_fun(x)
        x = x @ self.coeffs
        x = F.tanh(x)
        return x


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


    
class RBFRFFNet(nn.Module):
    
    def __init__(self, name, dim_mpl_layers, f_nfeatures, f_scale, n_centers, n_out):
        super(RBFRFFNet, self).__init__()
        self.name = name
        
        # regression/pinn network       
        self.rff = Fourier(f_nfeatures, f_scale) # directly the random matrix 'cause of checkpoint and load
        self.mlp = MLP(dim_mpl_layers)
        self.rbf = RBFNet(dim_mpl_layers[-1], n_out, n_centers)
    
    def forward(self, x): # x := BxC(Batch, InputChannels)
        ## implement periodicity
        x = torch.remainder(x,1)
        ## Fourier features
        x = self.rff(x) # Batch x Fourier Features
        ## MLP
        x = self.mlp(x)
        ## DivFree
        x = self.rbf(x)
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
                y_hat = self.forward(x_batch)
                loss = F.mse_loss(y_hat, y_batch)
                current_loss += (1/batches) * (loss.item() - current_loss)
                loss.backward()
                optimiser.step()
                progress += y_batch.size(0)
                if epoch % 100 == 0:
                    print('Epoch: %d, Loss: %f' % (epoch, current_loss))
        print('Done with Training')
        print('Final error:', current_loss)