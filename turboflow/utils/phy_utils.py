import torch
import numpy as np
import matplotlib.pyplot as plt

"""
Sources
- https://github.com/maxjiang93/space_time_pde
- https://github.com/Rose-STL-Lab/Turbulent-Flow-Net
- https://github.com/b-fg/Energy_spectra
"""

def _check_dims_img(uv):
    # check that uv is (R,R,2)
    res = uv.shape[1:]
    assert len(res) == uv.shape[0] == 2
    assert res[0] == res[1]

def _check_dim_vect(uv):
    # check that uv is (R*R,2)
    assert len(uv.shape) == 2
    assert uv.shape[1] == 2


def my_grad(f:tuple, sp:tuple, indexing:str = "xy"):
    """
    my local computation of the gradient
    
    f: tuple 
        -> vector field components [Fx,Fy,Fz,...]
    sp: tuple 
        -> spacing between points in respecitve directions [spx, spy,spz,...]
        -> or 1N array for the coordinates [x,y,z,...]
    indexing: str 
        "xy" or "ij", see np.meshgrid indexing 
    
    Returns:
        Components x Directions: grad[j][i] is the gradient of j-th component of F with respect direction i
    """
    num_dims_f = len(f)
    num_dims_sp = len(sp)
    if indexing == "xy":
        raise NotImplementedError
        # return [[np.gradient(f[num_dims - j - 1], sp[i], axis=i, edge_order=1) 
        #         for i in range(num_dims)] for j in range(num_dims)]
    if indexing == "ij":
        return [[np.gradient(f[j], sp[i], axis=i, edge_order=1) 
                for i in range(num_dims_sp)] for j in range(num_dims_f)]


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


def powerspec(uv):
    res = uv.shape[-2:]

    assert len(res) == uv.shape[0]
    assert res[0] == res[1]

    eps = 1e-50 # to avoid log(0)
    c  = np.sqrt(1.4)
    Ma = 0.1
    U0 = Ma*c
    U = uv/U0
    dims = uv.shape[1] * uv.shape[2]
    
    amplsU = abs(np.fft.fft2(U[0,:,:])/dims)
    amplsV = abs(np.fft.fft2(U[1,:,:])/dims)

    EK_U  = 0.5*(amplsU**2 + amplsV**2)

    EK_U = np.fft.fftshift(EK_U)

    sign_sizex = np.shape(EK_U)[0]
    sign_sizey = np.shape(EK_U)[1]

    box_sidex = sign_sizex
    box_sidey = sign_sizey

    box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)

    centerx = int(box_sidex/2)
    centery = int(box_sidey/2)
                
    EK_U_avsphr = np.zeros(box_radius,)+eps ## size of the radius
    

    for i in range(box_sidex):
        for j in range(box_sidey):
            wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2)))
            EK_U_avsphr[wn] = EK_U_avsphr[wn] + EK_U[i,j]

    k_bin = np.arange(0,box_radius)
    return EK_U_avsphr, k_bin


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


def energy_spectrum(uv):
    """
    Compute energy spectrum given a velocity field
    :param vel: tensor of shape (2, R, R)
    :return spec: tensor of shape(res/2)
    :return k: tensor of shape (res/2,), frequencies corresponding to spec
    """
    res = uv.shape[-2:]

    assert len(res) == uv.shape[0]
    assert res[0] == res[1]

    eps = 1e-50
    c = np.sqrt(1.4)
    Ma= 0.1
    U0 = Ma * c
    U = uv/U0
    dims = uv.shape[1] * uv.shape[2]

    Ek_u = torch.abs(torch.fft.fft2(uv[0,:,:])/dims).float()
    Ek_v = torch.abs(torch.fft.fft2(uv[1,:,:])/dims).float()
    Ek = 0.5 * (Ek_u**2 + Ek_v**2)
    Ek = torch.fft.fftshift(Ek)

    box_size_x = Ek.shape[0]
    box_size_y = Ek.shape[1]
    
    cx = int(box_size_x/2)
    cy = int(box_size_y/2)

    box_radius = int(np.ceil((np.sqrt((box_size_x)**2+(box_size_y)**2))/2.)+1)
    
    EK_U_avsphr = torch.zeros(box_radius).float().to(uv.device) + eps

    for i in range(box_size_x):
        for j in range(box_size_y):
            # EK_U_avsphr[wn[i,j]] = EK_U_avsphr[wn[i,j]] + Ek[i,j]
            wn =  int(np.round(np.sqrt((i-cx)**2+(j-cy)**2)))
            EK_U_avsphr[wn] = EK_U_avsphr[wn] + Ek[i,j]
    
    spec = EK_U_avsphr
    k_bin = torch.arange(0,len(spec))
    return spec, k_bin

def fast_energy_spectrum(uv):
    """
    Compute energy spectrum given a velocity field
    :param vel: tensor of shape (2, R, R)
    :return spec: tensor of shape(res/2)
    :return k: tensor of shape (res/2,), frequencies corresponding to spec
    """
    res = uv.shape[-2:]

    assert len(res) == uv.shape[0]
    assert res[0] == res[1]

    eps = 1e-50
    c = np.sqrt(1.4)
    Ma= 0.1
    U0 = Ma * c
    U = uv/U0
    dims = uv.shape[1] * uv.shape[2]

    Ek_u = torch.abs(torch.fft.fft2(uv[0,:,:])/dims).float()
    Ek_v = torch.abs(torch.fft.fft2(uv[1,:,:])/dims).float()
    Ek = 0.5 * (Ek_u**2 + Ek_v**2)
    Ek = torch.fft.fftshift(Ek)

    box_size_x = Ek.shape[0]
    box_size_y = Ek.shape[1]
    
    cx = int(box_size_x/2)
    cy = int(box_size_y/2)

    box_radius = int(np.ceil((np.sqrt((box_size_x)**2+(box_size_y)**2))/2.)+1)
    
    EK_U_avsphr = torch.zeros(box_radius).float().to(uv.device) + eps

    i = torch.arange(box_size_x).long()
    j = torch.arange(box_size_y).long()
    ii, ij = torch.meshgrid(i, j)
    wn = torch.round(torch.sqrt((ii - cx)**2 + (ij - cy)**2)).long()
    print(wn.shape)
    print(Ek.shape)
    print(Ek[wn].shape)

    bins = torch.arange(0,box_radius)
    inds = torch.searchsorted(bins, Ek.flatten())
    
    1/0

    for i in range(box_size_x):
        for j in range(box_size_y):
            # EK_U_avsphr[wn[i,j]] = EK_U_avsphr[wn[i,j]] + Ek[i,j]
            wn =  int(np.round(np.sqrt((i-cx)**2+(j-cy)**2)))
            EK_U_avsphr[wn] = EK_U_avsphr[wn] + Ek[i,j]
    
    spec = EK_U_avsphr
    k_bin = torch.arange(0,len(spec))
    return spec, k_bin


def energy_spectrum_bfg(uv, xygrid=[0,1,0,1], kres=32):
    """
    Return kinetic energy (KE) of a two-dimensional velocity vector field in Fourier space (uk, vk)
    given the wavenumber vector (kx, ky).
    First, the kinetic energy is computed at each possible wavenumber in Fourier space and stored
    in the E_entries[total number of entries, columns for k and E] array. Note that the wavenumber
    kmod might be repeated during the procedure, so this is why the energy is stored as a set of entries.
    Then, a loop through the entries stores the energy at the corresponging wavenumber bandwidth
    kmod - dk/2 <= kmod <= kmod + dk/2, where dk is calculated acording to the desired resolution kres.
    """

    device = uv.device
    res = uv.shape[-2:]

    assert len(res) == uv.shape[0]
    assert res[0] == res[1]

    N, M = res
    uk = torch.fft.fft2(uv[0,:,:])/(N*M)
    vk = torch.fft.fft2(uv[1,:,:])/(N*M)

    xmin, xmax = xygrid[0], xygrid[1]
    ymin, ymax = xygrid[2], xygrid[3]

    alpha = 2*np.pi/(xmax-xmin)
    beta  = 2*np.pi/(ymax-ymin)
    # Calculate the index per wavenumber direction: eg x: 0,1,2,...,N/2-1,-N/2,-N/2+1,...,-1
    x_index = torch.fft.fftfreq(N, d=1/N)
    y_index = torch.fft.fftfreq(M, d=1/M)
    # Initialize arrays
    kx = torch.zeros(N)
    ky = torch.zeros(M)
    for i in range(0, N):
        kx[i] = alpha*x_index[i]
    for j in range(0, M):
        ky[j] = beta*y_index[j]

    # Calculate the E_entries array
    E_entries = torch.zeros((N*M, 2))
    for i in range(0, N):
        for j in range(0, M):
            kmod = torch.sqrt(kx[i]**2+ky[j]**2)
            a = uk[i, j]*uk[i, j].conj()
            b = vk[i, j]*vk[i, j].conj()
            E = 0.5*(a.real+b.real)
            E_entries[i*M+j, 0] = kmod
            E_entries[i*M+j, 1] = E

    # Integrate the energy corresponding to a certain bandwidth dk
    kmin = 0
    kmax = torch.sqrt(torch.max(kx)**2+torch.max(ky)**2)
    dk = (kmax-kmin)/kres
    
    KE = torch.zeros((kres, 2))
    KE[:, 0] = torch.linspace(0, kres-1, kres)*dk+dk/2  # k values at half of each bandwidth
    
    for i in range(0, N*M):
        kmod = E_entries[i, 0]
        kint = int(kmod/dk)
        if kint >= kres:
            KE[-1, 1] = KE[-1, 1] + E_entries[i, 1]
        else:
            KE[kint, 1] = KE[kint, 1] + E_entries[i, 1]

    return KE[:,1], KE[:,0]


def _my_spec_grad(S):
    """
    Compute spectral gradient of a scalar field
    """
    assert(len(S.shape) in [3])
    res = S.shape[-1]
    # compute the frequency support
    k = torch.fft.fftfreq(res, d=1/res, device=S.device)
    k = torch.stack(torch.meshgrid([k, k]), dim=0)
    # compute gradient (dS = j*w*S)
    return 1j*k*S

def fluct(uv):
    _check_dims_img(uv)
    return uv - torch.mean(uv, dim=[0,1])

def tkenergy(uv):
    _check_dims_img(uv)
    # tke = 0.5 * (uv[0,:,:]**2 + uv[1,:,:]**2)
    tke = 0.5 * torch.mean(uv**2, dim=0)
    return torch.mean(tke)


def dissipation(uv, viscosity):
    _check_dims_img(uv)
    UV = torch.fft.fft2(uv) # (2, Rx, Ry)
    dUV = _my_spec_grad(UV) # (2, Rx, Rx)
    dUVt = dUV.permute(0,2,1) # (2, Ry, Rx)
    # compute (spectral) strain
    S = 0.5 * (dUV + dUVt)
    s = torch.fft.ifft2(uv).real
    diss = 2 * viscosity * torch.mean(s**2, dim=0)
    return torch.mean(diss)


def rmsvelocity(uv):
    _check_dims_img(uv)
    return tkenergy(uv) * (2./3.)**(1./2.)


def tmscale(uv, viscosity):
    rmvs = rmsvelocity(uv)
    diss = dissipation(uv, viscosity=viscosity)
    return (15*viscosity*(rmvs**2)/diss)**(1/2)


def tsreynolds(uv, viscosity):
    rmsv = rmsvelocity(uv)
    lam = tmscale(uv, viscosity)
    return rmsv * lam / viscosity


def ktimescale(uv, viscosity):
    diss = dissipation(uv, viscosity)
    return (viscosity/diss)**(1./2.)


def klenscale(uv, viscosity):
    diss = dissipation(uv, viscosity)
    return viscosity**(3/4) * diss**(-1/4)


def intscale(uv):
    UV, k = energy_spectrum(uv)
    rmsv = rmsvelocity(uv)

    c1 = np.pi/(2*rmsv**2)
    c2 = torch.sum(UV / k, dim=0)
    return c1 * c2
