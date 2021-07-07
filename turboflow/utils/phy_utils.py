import numpy as np
import matplotlib.pyplot as plt

def compute_vorticity(U, V):
    print('Warning *** CHECK THE GRADIENT DIRECTION')
    dUx, dUy = np.gradient(U)
    dVx, dVy = np.gradient(V)
    return dVx - dUy
    
def compute_divergence(U, V):
    print('Warning *** CHECK THE GRADIENT DIRECTION')
    dUx, dUy = np.gradient(U)
    dVx, dVy = np.gradient(V)
    return dUx + dVy

def compute_magnitude(U, V):
    return np.sqrt(U**2 + V**2)


def plot_field(U,V,scale=5,step=20,img=None, ax=None):
    """
    Created on Tue Sep 15 13:22:23 2015​
    @author: corpetti

    affichage d'un champ de vecteurs
    Input : u,v,scale,step,image (3 derniers optionnels)
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    if img is None:
        ax1 = ax.quiver(scale*np.flipud(U[::step,::step]),scale*np.flipud(-V[::step,::step]))
        return ax1
    else:
        ny,nx = img.shape
        X,Y = np.meshgrid(np.arange(0,nx,step), np.arange(0,ny,step))            
        ax1 = ax.quiver(X,Y,U[::step,::step], -V[::step,::step], color='w')
        ax2 = ax.imshow(img,cmap='gray')
        return [ax1, ax2]

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