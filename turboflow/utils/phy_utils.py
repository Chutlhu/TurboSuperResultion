import numpy as np
import matplotlib.pyplot as plt


def my_grad(f, sp, indexing = "xy"):
    num_dims = len(f)
    if indexing == "xy":
        return [[np.gradient(f[num_dims - j - 1], sp[i], axis=i, edge_order=1) 
                for i in range(num_dims)] for j in range(num_dims)]
    if indexing == "ij":
        return [[np.gradient(f[j], sp[i], axis=i, edge_order=1) 
                for i in range(num_dims)] for j in range(num_dims)]

# def compute_vorticity(u, v):
#     dUx, dUy = np.gradient(u)
#     dVx, dVy = np.gradient(v)
#     return dVx - dUy
    
# def compute_divergence(U, V):
#     dUx, dUy = np.gradient(U)
#     dVx, dVy = np.gradient(V)
#     return dUx + dVy

def compute_vorticity(xy:tuple, uv:tuple, indexing='ij'):
    x = xy[0][:,0]
    y = xy[1][0,:]
    du_xy = my_grad([uv[0], uv[1]], [x, y], indexing=indexing)
    return du_xy[1][0] - du_xy[0][1]

def compute_divergence(xy:tuple, uv:tuple, indexing='ij'):
    
    x = xy[0][:,0]
    y = xy[1][0,:]

    du_xy = my_grad([uv[0], uv[1]], [x, y], indexing=indexing)
    return du_xy[0][0] + du_xy[1][1]


def compute_magnitude(U, V):
    return np.sqrt(U**2 + V**2)


def divergence(f, sp, indexing = "xy"):
    """ 
    Computes divergence of vector field 
    f: array -> vector field components [Fx,Fy,Fz,...]
    sp: array -> spacing between points in respecitve directions [spx, spy,spz,...]
    indexing: "xy" or "ij", see np.meshgrid indexing 

    """
    num_dims = len(f)
    
    if indexing == "xy":
        return np.ufunc.reduce(np.add, [np.gradient(f[num_dims - i - 1], sp[i], axis=i) for i in range(num_dims)])
    if indexing == "ij":
        return np.ufunc.reduce(np.add, [np.gradient(f[i], sp[i], axis=i) for i in range(num_dims)])


def powerspec(data):
    # print ("shape of data = ",data.shape)
    eps = 1e-50 # to avoid log(0)
    c  = np.sqrt(1.4);
    Ma = 0.1;
    U0 = Ma*c; 
    U = data/U0
    amplsU = abs(np.fft.fftn(U)/U.size)

    EK_U  = amplsU**2

    EK_U = np.fft.fftshift(EK_U)

    sign_sizex = np.shape(EK_U)[0]
    sign_sizey = np.shape(EK_U)[1]

    box_sidex = sign_sizex
    box_sidey = sign_sizey

    box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)

    centerx = int(box_sidex/2)
    centery = int(box_sidey/2)

    # print ("box sidex     =",box_sidex) 
    # print ("box sidey     =",box_sidey) 
    # print ("sphere radius =",box_radius )
    # print ("centerbox     =",centerx)
    # print ("centerboy     =",centery)
                
    EK_U_avsphr = np.zeros(box_radius,)+eps ## size of the radius1234
    

    for i in range(box_sidex):
        for j in range(box_sidey):
            wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2)))
            EK_U_avsphr[wn] = EK_U_avsphr [wn] + EK_U [i,j]

    return EK_U_avsphr


def plot_energy_spec(K, ax=None, label=''):

    if ax is None:
        fig = plt.figure()
        plt.title("Kinetic Energy Spectrum")
        plt.xlabel(r"k (wavenumber)")
        plt.ylabel(r"TKE of the k$^{th}$ wavenumber")
        
        realsize=len(np.fft.rfft(K))
        # plt.plot(K)
        plt.loglog(np.arange(0,realsize),((K[0:realsize] )),'k', label=label)
        # plt.loglog(np.arange(realsize,len(K),1),((K[realsize:] )),'k--')    
        
        ax = plt.gca()
        ax.set_ylim([10**-25,10])

    else:

        ax.set_xlabel(r"k (wavenumber)")
        ax.set_ylabel(r"TKE of the k$^{th}$ wavenumber")

        realsize=len(np.fft.rfft(K))
        ax.plot(K)
        ax.loglog(np.arange(0,realsize),((K[0:realsize] )),'k')
        ax.loglog(np.arange(realsize,len(K),1),((K[realsize:] )),'k--')    
        
        ax.set_ylim([10**-25,10])

    return ax


def compute_stucture_function(u, direction='h'):
    # u must be HxWx2
    assert len(u.shape) == 3 
    assert u[2] == 2

    sfun2 = lambda x, xd : (x - xd)**2
    sfun3 = lambda x, xd : (x - xd)**3

    if direction == 'u':
        pass

    return 



