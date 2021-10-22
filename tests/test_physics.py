import pytest
import numpy as np

import turboflow.utils.phy_utils as phy

def local_fun1(xx, yy):
    # F = [ cos(x+2*y), sin(x-2*y) ]
    return {
        'uu' : np.cos(xx+2*yy),
        'vv' : np.sin(xx-2*yy),
        'du_dx' :   -np.sin(xx+2*yy),
        'du_dy' : -2*np.sin(xx+2*yy),
        'dv_dx' :    np.cos(xx-2*yy),
        'dv_dy' : -2*np.cos(xx-2*yy),
    }


def test_derivate_ij():
    x = np.linspace(0, 0.5, 1000)
    y = np.linspace(0, 0.5, 1000)

    yy, xx = np.meshgrid(x, y) # reverse output to have ij indexing

    f = local_fun1(xx, yy)

    grad = phy.my_grad((f['uu'], f['vv']),(x,y), indexing='ij')

    assert np.allclose(grad[0][0][3:-3,3:-3], f['du_dx'][3:-3,3:-3], atol=1e-6)
    assert np.allclose(grad[0][1][3:-3,3:-3], f['du_dy'][3:-3,3:-3], atol=1e-6)
    assert np.allclose(grad[1][0][3:-3,3:-3], f['dv_dx'][3:-3,3:-3], atol=1e-6)
    assert np.allclose(grad[1][1][3:-3,3:-3], f['dv_dy'][3:-3,3:-3], atol=1e-6)

def test_divergence():
    x = np.linspace(0, 0.5, 1000)
    y = np.linspace(0, 0.5, 1000)
    yy, xx = np.meshgrid(x, y)
    
    f = local_fun1(xx, yy)

    d = phy.compute_divergence((xx, yy), (f['uu'], f['vv']), indexing='ij')
    d_hc = f['du_dx'] + f['dv_dy']
    assert np.allclose(d[3:-3,3:-3], d_hc[3:-3,3:-3], atol=1e-6)
    
def test_vorticity():
    x = np.linspace(0, 0.5, 1000)
    y = np.linspace(0, 0.5, 1000)
    yy, xx = np.meshgrid(x, y)
    
    f = local_fun1(xx, yy)

    w = phy.compute_vorticity((xx, yy), (f['uu'], f['vv']), indexing='ij')
    w_hc = f['dv_dx'] - f['du_dy']
    assert np.allclose(w[3:-3,3:-3], w_hc[3:-3,3:-3], atol=1e-6)

def test_raise_error_derivate_xy():
    x = np.linspace(0, 0.5, 1000)
    y = np.linspace(0, 0.5, 1000)

    xx, yy = np.meshgrid(x, y)
    uu = np.cos(xx+2*yy)
    vv = np.sin(xx-2*yy)
    
    with pytest.raises(NotImplementedError):
        grad = phy.my_grad((uu,vv),(x,y), indexing='xy')