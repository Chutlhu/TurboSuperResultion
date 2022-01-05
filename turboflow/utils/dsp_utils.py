import numpy as np
from scipy import interpolate

from turboflow.utils import alg_utils as alg

'''
xlr = Xlr[:,0].reshape(L,L)[:,0]
ylr = Xlr[:,1].reshape(L,L)[0,:]

print(xlr)
print(ylr)
'''

def interpolate2D_mesh01x01(xy, z, scale):

    assert xy.shape[0] == xy.shape[1]
    assert xy.shape[2] == 2

    in_npoints = xy.shape[0]
    out_npoints = int(in_npoints * scale)

    x = xy[:,0,0]
    y = xy[0,:,1]
    z = z

    assert np.allclose(np.diff(np.diff(x)), 0)
    assert np.allclose(np.diff(np.diff(y)), 0)
    delta_x = x[1] - x[0]
    delta_y = y[1] - y[0]

    f_interp2d = interpolate.interp2d(x, y, z, kind='cubic') # 'cubic'

    x = np.linspace(0, 1, out_npoints, endpoint=True)
    y = np.linspace(0, 1, out_npoints, endpoint=True)

    assert x[-1] <= 1.
    assert y[-1] <= 1.

    xy = np.stack(np.meshgrid(x, y), axis=-1)


    return f_interp2d(x, y), xy

def interpolate2D_t(ti, xi, yi, f_txyi, to, xo, yo):

    assert len(xi) > 1
    assert len(yi) > 1

    if ti is None:
        in_points = (xi, yi)         # input grid
        values = f_txyi
        out_points = alg.make_xy_grid(xo, yo)

        interpolation = interpolate.interpn(in_points, values, out_points)[None,...]

    if len(ti) == 1:
        assert len(ti) == len(to) == f_txyi.shape[0]

        in_points = (xi, yi)         # input grid
        values = f_txyi[0,:,:]
        out_points = alg.make_xy_grid(xo, yo)

        interpolation = interpolate.interpn(in_points, values, out_points)[None,...]

    else:
        in_points = (ti, xi, yi)         # input grid
        values = f_txyi
        out_points = alg.make_xyt_grid(to, xo, yo)

        interpolation = interpolate.interpn(in_points, values, out_points)

    return interpolation