import torch
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.fftpack import fft, dct

from argparse import Namespace
from tqdm import tqdm

print(torch.cuda.is_available())
device = 'cuda'
device = 'cpu'


x, fs = sf.read('./data/piano.wav')
# to mono
x = np.mean(x, axis=-1)

dur = 0.08

x = x[2*fs:int((2+dur)*fs)]
x = x - np.mean(x)
x = x / np.max(np.abs(x))
t = np.arange(0,dur,1/fs)

# Discrete coSine Transform
X = dct(x)[:len(x)//4]
X = X/np.max(np.abs(X))
f = np.linspace(0,1,len(X))
# plt.plot(f, X)
# plt.show()

s = torch.from_numpy(x).float().to(device)
t = torch.from_numpy(t).float().to(device)

# freq domains
S = torch.from_numpy(X).float().to(device)
f = torch.from_numpy(f).float().to(device)

class CurrDataset(torch.utils.data.Dataset):

    def __init__(self, t, x):
        super(CurrDataset, self).__init__()

        self.x = x[:,None]
        self.t = t[:,None]
        assert x.shape == t.shape
        self.size = x.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        t = self.t[idx,:]
        x = self.x[idx,:]
        return t, x

dataset = CurrDataset(t, s)
invar, outvar = dataset[:]

plt.plot(invar, outvar)
plt.savefig('./figures/out_vs_in.png')

from model import iMFN

dataloader = torch.utils.data.DataLoader(dataset, batch_size=267, shuffle=True)
model = iMFN()
model.to(device)
print(model)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


#initialize empty list to track batch losses
batch_losses = []

kickin = 1000

#train the neural network for 5 epochs
loss_v = 1
lam_v = 1
lam_w = 0
loss_w = 0
loss_v = 0
for epoch in tqdm(range(3000)):

    #reset iterator
    batch = iter(dataloader)
    
    for batch_idx, (invar, outvar) in enumerate(batch):
                
        #forward propagation through the network
        
        #reset gradients
        optimizer.zero_grad()

        outvar_pred, w, wn = model(invar, stage='train_time')
        loss_v = F.mse_loss(outvar_pred, outvar)
        if loss_v > 0.003:
            if batch_idx == 0 and epoch % 30 == 0: print('Train TIME')
            outvar_pred, w, wn = model(invar, stage='train_time')
            lam_v = 1
            lam_w = 1e-3
        else:
            if batch_idx == 0 and epoch % 30 == 0: print('Train FREQ')
            outvar_pred, w, wn = model(invar, stage='train_freq')
            lam_v = 0
            lam_w = 1e3
        
        #calculate the loss
        loss_v = F.mse_loss(outvar_pred, outvar)
        assert wn.shape == w.T.shape
        # loss_w_l1 = F.mse_loss(wn, w.T)
        loss_w_l1 = torch.mean(torch.abs(wn - w.T), dim=0)
        loss_w_cs = 1 - (wn * w.T) / (torch.abs(wn) * torch.abs(w.T))
        loss_w_cs = torch.mean(loss_w_cs, dim=0)
        loss_w = 100*loss_w_l1 + loss_w_cs

        loss = lam_v * loss_v + lam_w*loss_w

        #track batch loss
        batch_losses.append(loss.item())

        #backpropagation
        loss.backward()

        #update the parameters
        optimizer.step()
    
    if epoch % 20 == 0:
        scheduler.step()
        
        
    if epoch % 30 == 0:
        print(f'Loss var {loss_v} | loss weights {loss_w.item()}')
        # print(outvar_pred[:8,0].data)

    if loss < 1e-4: 
        print('Loss less the 5e-4: ending training.')
        break

torch.save(model.state_dict(), './recipes/audio/model.ptc')

