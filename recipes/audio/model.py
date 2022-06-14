from turtle import shape
import torch
import torch.nn as nn
import numpy as np

class GaborFilter(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta=1.0, max_freq=128.):
        super(GaborFilter, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim)

        # Init weights
        self.linear.weight.data *= max_freq * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        norm = (x ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(x))


class FourierFilter(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta=2.0, bias=True, max_freq=128.):
        super(FourierFilter, self).__init__()

        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # Init weights
        self.linear.weight.data *= max_freq * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return torch.sin(self.linear(x))

class MFN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, k=4, filter_fun='Gabor', data_max_freq=128):
        super(MFN, self).__init__()

        if filter_fun == 'Gabor':
            filter_fun = GaborFilter
        else:
            filter_fun = FourierFilter

        self.k = k
        self.filters = nn.ModuleList(
            [filter_fun(in_dim, hidden_dim, alpha=6.0 / k, max_freq=data_max_freq[i]) for i in range(k)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(k - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])
        
        # self.noise_filters = nn.ModuleList(
        #     [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(k - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])

        for lin in self.linear[:k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim))

        # for lin in self.noise_filters[:k - 1]:
        #     lin.weight.data.constant_(1.)

    def forward(self, x):
        # Recursion - Equation 3
        zi = self.filters[0](x)  # Eq 3.a
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.filters[i + 1](x)  # Eq 3.b
        x = self.linear[self.k - 1](zi)  # Eq 3.c
        return x

class FourierBasis(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta=2.0, bias=True, max_freq=128.):
        super(FourierBasis, self).__init__()

        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
        
        self.max_freq = max_freq

        # Init weights
        self.linear.weight.data *= max_freq * torch.sqrt(self.gamma.unsqueeze(-1))
        if bias:
            self.linear.bias.data.uniform_(-np.pi, np.pi)
        
#         B = torch.linspace(0, max_freq, 1024)[:,None]
#         self.B = nn.Parameter(B, requires_grad=False)
        
        
    def forward(self, x):
        ''' x in B x 1
            z in F x 1
        '''
        x = self.linear(x)
        return torch.sin(x) + torch.cos(x)


class ImplicitFourierFilter(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta=2.0, bias=True, max_freq=128.):
        super(ImplicitFourierFilter, self).__init__()

        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
        
        self.max_freq = max_freq

        # Init weights
        self.linear.weight.data *= max_freq * torch.sqrt(self.gamma.unsqueeze(-1))
        if bias:
            self.linear.bias.data.uniform_(-np.pi, np.pi)
        
#         B = torch.linspace(0, max_freq, 1024)[:,None]
#         self.B = nn.Parameter(B, requires_grad=False)
#         
    def forward(self, x):
        ''' x in B x 1
            z in F x 1
        '''
        x = self.linear(x)
        return torch.sin(x)




class iMFN(nn.Module):
    def __init__(self):
        super(iMFN, self).__init__()

        K = 2
        
        # freqs to weights
        hidden_dim = 1024
        max_freq = 1e3
        self.fft = nn.Sequential(
            ImplicitFourierFilter(1, hidden_dim, alpha=6.0 / K, max_freq=max_freq, bias=True),
            torch.nn.Linear(hidden_dim, 256, bias=True),
            ImplicitFourierFilter(256, 256, alpha=6.0 / K, max_freq=max_freq, bias=True),
            torch.nn.Linear(256, 1, bias=False),
            # ImplicitFourierFilter(256, 64, alpha=6.0 / K, max_freq=max_freq, bias=True),
            # torch.nn.Linear(64, 1, bias=False),
        )
        
        # time to signal
        hidden_dim = 1024
        max_freq = 8e3 # 22e3

        self.time = nn.Sequential(
            FourierBasis(1, hidden_dim, alpha=6.0 / K, max_freq=max_freq, bias=False),
            torch.nn.Linear(hidden_dim, 1, bias=False),
        )
        self.time[1].weight.data.uniform_(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim))


    def forward(self, t, ftest=None, stage='train_time'):
        
        if stage == 'train_freq':
            self.fft.requires_grad_(True)
            self.time.requires_grad_(False)
        elif stage == 'train_time':
            self.fft.requires_grad_(True)
            self.time.requires_grad_(True)
       
        f = self.time[0].linear.weight.data # target freqs NFFTx1
        w = self.time[1].weight.data # target weights 1xNFFT
        wn = w.T
        if stage == 'train_freq':
            # random selection and shuffle of f, w, and wn
            nfft = w.shape[1]
            idx = torch.randint(256, nfft, size=[nfft])
            f = f[idx,:]
            w = w[:,idx]
            wn = self.fft(f) #NFFTx1

        if stage in ['train_time', 'train_freq']:
            x = self.time(t) # B x out
        
        if stage == 'test':
            wn = self.fft(ftest)
            self.time[0].linear.weight.data = ftest
            self.time[1].weight.data = wn.T
            x = self.time(t)
            
        return x, w, wn


class moMFN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, k=4, filter_fun='Gabor', data_max_freq=128):
        super(moMFN, self).__init__()

        if filter_fun == 'Gabor':
            filter_fun = GaborFilter
        else:
            filter_fun = FourierFilter

        self.k = k
        self.filters = nn.ModuleList(
            [filter_fun(in_dim, hidden_dim, alpha=6.0 / k, max_freq=data_max_freq[i]) for i in range(k)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(k - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])
        
        self.mid_linear = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(k - 1)])

        for lin in self.linear[:k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim))

        # for lin in self.noise_filters[:k - 1]:
        #     lin.weight.data.constant_(1.)

    def forward(self, x):
        # Recursion - Equation 3
        zi = self.filters[0](x)  # Eq 3.a
        ys = []
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.filters[i + 1](x)  # Eq 3.b
            yi = self.mid_linear[i](zi)
            ys.append(yi)

        x = self.linear[self.k - 1](zi)  # Eq 3.c
        
        return x, ys