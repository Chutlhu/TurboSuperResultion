import numpy as np
from scipy import interpolate


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


