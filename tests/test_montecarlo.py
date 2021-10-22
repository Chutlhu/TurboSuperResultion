import torch
import pytest

from turboflow.models.basics import montecarlo_sampling_xcenters_xincerments

def test_montecarlo_sampling():
    n_points = 3
    n_increments = 5
    min_l = 0.3
    device = 'cpu'

    P, Pt = montecarlo_sampling_xcenters_xincerments(n_points, n_increments, min_l, device)

    assert P.shape[0] == n_increments
    assert P.shape[1] == n_points
    assert P.shape[2] == 2

    r = torch.sqrt(torch.sum((Pt-P)**2, dim=2))

    assert torch.allclose(r[:,0],r.mean(dim=1))
    assert torch.allclose(r[:,0], (torch.arange(n_increments)*min_l).to(float))
