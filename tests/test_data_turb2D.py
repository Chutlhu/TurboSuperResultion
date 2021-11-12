import pytest
import pathlib
import numpy as np

import torch
from torch.utils.data import DataLoader

from turboflow.datasets.turb2D import Turb2D
from turboflow.dataloaders import Turb2DDataset

data_dir = pathlib.Path('/','home','dicarlo_d','Documents','Datasets','2021-Turb2D')
tb = Turb2D(data_dir)
tb.setup()

# TEST TURB2D
def test_turb2D_single_time_idx():
    t = np.random.randint(1000)
    tb.load_data(t)

    X = tb.xy
    y = tb.uv

    assert len(X.shape) == 3
    assert X.shape == y.shape
    assert X.shape[0] == y.shape[0] == 256
    assert X.shape[1] == y.shape[1] == 256
    assert X.shape[2] == y.shape[2] == 2

    assert np.allclose(tb.xy[:,:,0], tb.x)
    assert np.allclose(tb.xy[:,:,1], tb.y)
    assert np.allclose(tb.uv[:,:,0], tb.u)
    assert np.allclose(tb.uv[:,:,1], tb.v)


@pytest.mark.parametrize("time_idx", [1000, 1002])
def test_raise_error_turb2D_time_idx(time_idx):
    with pytest.raises(IndexError):
        tb.load_data(time_idx)


def test_turb2D_multiple_time_indeces():
    n = 10
    t = np.arange(10)
    tb.load_data(t)

    X = tb.xy
    y = tb.uv

    assert len(X.shape) == 4
    assert X.shape == y.shape
    assert X.shape[0] == y.shape[0] == n
    assert X.shape[1] == y.shape[1] == 256
    assert X.shape[2] == y.shape[2] == 256
    assert X.shape[3] == y.shape[3] == 2

    assert len(tb.t) == n
    assert np.allclose(tb.xy[:,:,:,0], tb.x)
    assert np.allclose(tb.xy[:,:,:,1], tb.y)
    assert np.allclose(tb.uv[:,:,:,0], tb.u)
    assert np.allclose(tb.uv[:,:,:,1], tb.v)


def test_raise_error_turb2D_multiple_time_indeces():
    with pytest.raises(IndexError):
        t = np.arange(999,1010)
        tb.load_data(t)


def test_turb2D_all_time_indeces():
    t = None
    tb.load_data(t)

    assert len(tb.t) == tb.nt == 1000
    assert tb.xy.shape == (1000,256,256,2)
    assert tb.uv.shape == (1000,256,256,2)
    assert np.allclose(tb.xy[:,:,:,0], tb.x)
    assert np.allclose(tb.xy[:,:,:,1], tb.y)
    assert np.allclose(tb.uv[:,:,:,0], tb.u)
    assert np.allclose(tb.uv[:,:,:,1], tb.v)

# # TEST DATALOADERS
def test_dataloader_turb2D_single_time_idx():
    t = np.random.randint(1000)
    trainset = Turb2DDataset(data_dir, ds=1, time_idx=t)
    X, y = trainset[:]
    assert X.shape == y.shape
    assert X.shape[1] == 2


def test_dataloader_turb2D_multiple_time_indeces():
    t = np.arange(200,300)
    trainset = Turb2DDataset(data_dir, ds=1, time_idx=t)
    X, y = trainset[:]
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 3
    assert X[0,0] == 2.

@pytest.mark.parametrize("ds", [1, 4, 16, 32])
def test_dataloader_turb2D_downsampling(ds):
    t = np.arange(200,300)
    trainset = Turb2DDataset(data_dir, ds=ds, time_idx=t)
    X, y = trainset[:]
    assert int((X.shape[0]/100)**.5) == 256//ds