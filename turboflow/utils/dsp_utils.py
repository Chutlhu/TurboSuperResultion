import numpy as np
from scipy import interpolate

from turboflow.utils import alg_utils as alg

def interpolate2D(xy, z, scale):
    assert xy.shape[0] == xy.shape[1]

    in_npoints = xy.shape[0]
    out_npoints = int(in_npoints * scale)

    x = np.linspace(0,1, in_npoints)
    y = np.linspace(0,1, in_npoints)
    z = z

    f_interp2d = interpolate.interp2d(x, y, z, kind='linear') # 'cubic'

    x = np.linspace(0, 1, out_npoints)
    y = np.linspace(0, 1, out_npoints)

    return f_interp2d(x, y)


def interpolate2D_t(ti, xi, yi, f_txyi, to, xo, yo):

    assert len(xi) > 1
    assert len(yi) > 1

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