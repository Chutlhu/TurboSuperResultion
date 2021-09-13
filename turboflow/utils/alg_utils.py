import numpy as np

def make_xyt_grid(axis_t, axis_x, axis_y):
    x, y, t = np.meshgrid(axis_x, axis_y, axis_t)
    x = x.transpose(2,0,1)
    y = y.transpose(2,0,1)
    t = t.transpose(2,0,1)

    mesh = np.stack([t,x,y], -1) # TXY x 3
    assert mesh.shape[-1] == 3

    # make x and y in the correct orientation
    tmp = mesh[:,:,:,1].copy()
    mesh[:,:,:,1] = mesh[:,:,:,2]
    mesh[:,:,:,2] = tmp

    return mesh

def make_xy_grid(axis_x, axis_y):
    x, y = np.meshgrid(axis_x, axis_y)

    mesh = np.stack([x,y], -1) # TXY x 2

    assert mesh.shape[-1] == 2

    # make x and y in the correct orientation
    tmp = mesh[:,:,0].copy()
    mesh[:,:,0] = mesh[:,:,1]
    mesh[:,:,0] = tmp

    return mesh
