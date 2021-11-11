import pytest
import pathlib
import numpy as np

import torch
from torch.utils.data import DataLoader

from turboflow.datasets.turb2D import Turb2D
from turboflow.dataloaders import Turb2DDataset
import turboflow.utils.phy_utils as phy

# TEST TURB2D
data_dir = pathlib.Path('/','home','dicarlo_d','Documents','Datasets','2021-Turb2D')
tb = Turb2D(data_dir)
tb.setup()
tb.load_data(1000)

X = tb.xy
y = tb.uv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X.to(device).float
y.to(device).float

R = int(X.shape[0]**.5)

D = 64

def test_data_dimension():
    print(X.shape)
    print(y.shape)
    assert False

# def test_energy_spec():
#     uv = X.view(R, R, 2)     # (R*R, 2)
#     uv = uv.permute(2, 0, 1) # (R, R, 2) -> (2, R, R)

#     Ek_torch, k = phy.energy_spectrum(uv)
#     Ek_numpy = phy.powerspec(uv[0,:,:].cpu().numpy())

#     assert np.allclose(Ek_torch.cpu().numpy, Ek_numpy)