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
tb.load_data(np.arange(20))

X = tb.xy
u = tb.uv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X.to(device).float
u.to(device).float

R = int(X.shape[0]**.5)

D = 64

