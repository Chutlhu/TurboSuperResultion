import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from numpy.core.fromnumeric import repeat

import turboflow.utils.phy_utils as phy

###############################################################################
##                                 SIMPLE PLOTS                              ##
###############################################################################

def plot_field(xy_img, uv_img, step=5, scale=20, vorticity_img=None, ax=None, indexing='ij'):
    """
    Created on Tue Sep 15 13:22:23 2015​
    @author: corpetti

    affichage d'un champ de vecteurs
    Input : u,v,scale,step,image (3 derniers optionnels)
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    s = step

    if indexing == 'xy':
        raise NotImplementedError

    if not vorticity_img is None:
        w = np.zeros_like(vorticity_img)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i,j] = vorticity_img[j, -i]

        xmin, xmax = np.min(xy_img[0][::s,::s]), np.max(xy_img[0][::s,::s])
        ymin, ymax = np.min(xy_img[1][::s,::s]), np.max(xy_img[1][::s,::s])
        ax.imshow(w,
                    extent=[xmin, xmax, ymin, ymax], 
                    origin=None, cmap='gray')
    
    color = np.sqrt((uv_img[0]**2) + (uv_img[1]**2))
    ax.quiver(
            xy_img[0][::s,::s],
            xy_img[1][::s,::s], 
            uv_img[0][::s,::s],
            uv_img[1][::s,::s], 
            # color = 'w',
            color[::s,::s], 
            scale=scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax

def animate_field(xy_img_t, uv_img_t, step=5, scale=20, indexing='ij'):
    
    assert len(xy_img_t) == len(uv_img_t) == 2
    assert xy_img_t[0].shape == xy_img_t[1].shape
    assert uv_img_t[0].shape == uv_img_t[1].shape
    assert xy_img_t[0].shape == uv_img_t[0].shape

    T = xy_img_t[0].shape[0]

    fig, ax = plt.subplots(figsize=(10,10))
    ims = []

    for t in range(T):

        xx = xy_img_t[0][t,...]
        yy = xy_img_t[1][t,...]
        uu = uv_img_t[0][t,...]
        vv = uv_img_t[1][t,...]

        w = phy.compute_vorticity((xx, yy), (uu, vv))

        im = plot_field((xx, yy), (uu, vv), w, step = step, scale=scale, ax=ax)

        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.close()

    return

###############################################################################


def plot_lr_hr_inset(ulr, uhr, L, H, title=None, 
region=(0.05, 0.35, 0.45, 0.75),
figsize=(6,6), add_extremes_in_title=False, only_u=False):


    if only_u:
        fig = plt.figure(figsize=figsize)
        
        i = 0
        IMG = ulr[:,i].reshape(L,L)
        IMGzoom = uhr[:,i].reshape(H,H)[:H,:H]
        
        extent = (0, 1, 0, 1)
        plt.imshow(IMG,  extent=extent, origin="upper")

        if add_extremes_in_title:
            suffix = ' in [%1.2f, %1.3f]' % (np.min(IMG[:,i]), np.max(IMG[:,i]))
        else:
            suffix = ''

        plt.title(title + r': $u_x$' + suffix)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')

        ax = plt.gca()

        # inset axes....
        axins = ax.inset_axes([0.4, 0.0, 0.6, 0.6])
        axins.imshow(IMGzoom, extent=extent, origin="upper")
        # sub region of the original image
        x1, x2, y1, y2 = region
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        ax.indicate_inset_zoom(axins, edgecolor="black")
    
    else:

        fig, axarr = plt.subplots(1,2,figsize=figsize)
        plt.suptitle(title,fontsize=20, y=0.75)

        for i in range(ulr.shape[-1]):
            IMG = ulr[:,i].reshape(L,L)
            IMGzoom = uhr[:,i].reshape(H,H)[:H,:H]
            
            extent = (0, 1, 0, 1)
            im = axarr[i].imshow(IMG,  extent=extent, origin="upper")

            if add_extremes_in_title:
                suffix = ' in [%1.2f, %1.3f]' % (np.min(IMG[:,i]), np.max(IMG[:,i]))
            else:
                suffix = ''

            if i == 0:
                axarr[i].set_title(r'$u_x$' + suffix)
            if i == 1:
                axarr[i].set_title(r'$u_y$' + suffix)
            axarr[i].set_xlabel(r'$x$')
            axarr[i].set_ylabel(r'$y$')

            # inset axes....
            axins = axarr[i].inset_axes([0.4, 0.0, 0.6, 0.6])
            axins.imshow(IMGzoom, extent=extent, origin="upper")
            # sub region of the original image
            x1, x2, y1, y2 = region
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels('')
            axins.set_yticklabels('')
            axarr[i].indicate_inset_zoom(axins, edgecolor="black")

    fig.tight_layout()

    return fig


def plot_velocity_field(xlr, ulr, L, xhr, uhr, H,
                        title=None,
                        region=(0.05, 0.35, 0.45, 0.75),
                        figsize=(10,10),
                        scale_lr=1, step_lr=1,
                        scale_hr=1, step_hr=1):

    xxlr = xlr[:,0].reshape(L,L)
    yylr = xlr[:,1].reshape(L,L)
    uxlr = ulr[:,0].reshape(L,L)
    uylr = ulr[:,1].reshape(L,L)
    color_lr = np.sqrt((uxlr**2) + (uylr**2))

    xxhr = xhr[:,0].reshape(H,H)
    yyhr = xhr[:,1].reshape(H,H)
    uxhr = uhr[:,0].reshape(H,H)
    uyhr = uhr[:,1].reshape(H,H)
    color_hr = np.sqrt((uxhr**2) + (uyhr**2))

    s = step_lr
    fig, ax = plt.subplots(1,1,figsize=figsize)
    plt.title(title)
    ax.quiver(xxlr[::s,::s], yylr[::s,::s], 
              uxlr[::s,::s], uylr[::s,::s], 
              color_lr[::s,::s], scale=scale_lr)
    plt.xlabel(r'$u_x$')
    plt.ylabel(r'$u_y$')

    # inset axes....
    s = step_hr
    axins = ax.inset_axes([0.4, 0.0, 0.6, 0.6])
    axins.quiver(xxhr[::s,::s], yyhr[::s,::s], 
                 uxhr[::s,::s], uyhr[::s,::s], 
                 color_hr[::s,::s], scale=scale_hr)
    # sub region of the original image
    x1, x2, y1, y2 = region
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()

    return fig
    

def plot_potential(xlr, Plr, L, xhr, Phr, H, title=None):

    fig = plt.figure(figsize=(15,10))
    fig.suptitle(title, fontsize=24, y=1)

    xxlr = xlr[:,0].reshape(L,L)
    yylr = xlr[:,1].reshape(L,L)
    zlr = Plr[:,0].reshape(L,L)

    xxhr = xhr[:,0].reshape(H,H)
    yyhr = xhr[:,1].reshape(H,H)
    zhr = Phr[:,0].reshape(H,H)

    extent = (0, 1, 0, 1)
    origin = 'upper'

    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(zlr, extent=extent, origin=origin)
    ax.set_title(r'LR Potential 2D $\Phi(\mathbf{x})$')
    ax.set_xlabel('x')
    ax.set_xlabel('y')

    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.plot_surface(xxlr, yylr, zlr, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title(r'LR Potential 3D $\Phi(\mathbf{x})$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\Phi(\mathbf{x})$')

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.plot_surface(xxlr[:L//3,:L//3], yylr[:L//3,:L//3], zlr[:L//3,:L//3],
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title(r'Zoom LR Potential 3D $\Phi(\mathbf{x})$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\Phi(\mathbf{x})$')

    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(zhr, extent=extent, origin=origin)
    ax.set_title(r'HR Potential 2D $\Phi(\mathbf{x})$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = fig.add_subplot(2, 3, 5, projection='3d')
    ax.plot_surface(xxhr, yyhr, zhr, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title(r'HR Potential 3D $\Phi(\mathbf{x})$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\Phi(\mathbf{x})$')

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.plot_surface(xxhr[:H//3,:H//3], yyhr[:H//3,:H//3], zhr[:H//3,:H//3], 
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title(r'Zoom HR Potential 3D $\Phi(\mathbf{x})$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\Phi(\mathbf{x})$')
    
    fig.tight_layout()
    return fig


def plot_model_losses(res_dict, title=None):

    fig = plt.figure()
    
    loss_rec = res_dict['loss']['rec']
    loss_pde = res_dict['loss']['pde']
    loss_reg = res_dict['loss']['reg']
    loss_sfn = res_dict['loss']['sfn']
    loss_tot = res_dict['loss']['tot']
    assert len(loss_pde) == len(loss_rec)

    epochs = np.arange(len(loss_rec))

    plt.plot(epochs, loss_rec, alpha=0.4, label=r'$\mathcal{L}_{rec}$')
    plt.plot(epochs, loss_pde, alpha=0.4, label=r'$\mathcal{L}_{pde}$')
    plt.plot(epochs, loss_sfn, alpha=0.4, label=r'$\mathcal{L}_{sfn}$')
    plt.plot(epochs, loss_reg, alpha=0.4, label=r'$\mathcal{L}_{reg}$')
    plt.plot(epochs, loss_tot, alpha=0.8, label=r'$\mathcal{L}_{tot}$')
    
    plt.yscale('log')
    plt.grid(True,which="both", linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Training Losses')

    plt.legend()
    plt.title(title)
    plt.tight_layout()

    return fig


def plot_energy_spectra(u_dicts, title=None):

    def energy_spectra_for_plotting(u, R):
        u = u.reshape(R,R,2)
        spectra_x = phy.powerspec(u[:,:,0])
        spectra_y = phy.powerspec(u[:,:,1])
        spectra = (spectra_x + spectra_y)/2
        support = np.arange(len(spectra))
        return spectra, support


    fig, axs = plt.subplots(1,1,figsize=(8,6))

    for n in range(len(u_dicts)):
        R = u_dicts[n]['size']
        u = u_dicts[n]['vel'].reshape(R,R,2)
        spectra, support = energy_spectra_for_plotting(u, R)
        label = u_dicts[n]['label']
        style = u_dicts[n]['style']
        axs.loglog(support, spectra, style, label=label)

    axs.set_ylim(10**(-13), 10)
    axs.set_xlim(1, 512)
    axs.legend()
    plt.ylabel('Energy specturm')
    plt.xlabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    return


###############################################################################
##                              WRAPPERS FOR RES_DICT                        ##
###############################################################################


def plot_potential_wrapper_res_dict(res_dict, title=None):

    xlr = res_dict['LR']['x']
    Plr = res_dict['LR']['P']
    L   = res_dict['LR']['size']

    xhr = res_dict['HR']['x']
    Phr = res_dict['HR']['P']
    H   = res_dict['HR']['size']

    return plot_potential(xlr, Plr, L, xhr, Phr, H, title=title)


def plot_lr_hr_inset_wrapper_res_dict(res_dict, title=None, add_extremes_in_title=False):

    ulr = res_dict['LR']['u']
    L   = res_dict['LR']['size']

    uhr = res_dict['HR']['u']
    H   = res_dict['HR']['size']

    return plot_lr_hr_inset(ulr, uhr, L, H, title=title, add_extremes_in_title=add_extremes_in_title)


def plot_lr_hr_inset_error_wrapper_res_dict(res_dict, title=None):
    ulr = res_dict['LR']['u']
    ulr_gt = res_dict['LR']['u_gt']
    L   = res_dict['LR']['size']

    uhr = res_dict['HR']['u']
    uhr_gt = res_dict['HR']['u_gt']
    H   = res_dict['HR']['size']

    err = lambda x, x_hat : np.abs(x-x_hat)

    return plot_lr_hr_inset(err(ulr, ulr_gt), err(uhr, uhr_gt), 
                            L, H, title=title, add_extremes_in_title=True)


def plot_velocity_field_wrapper_res_dict(
                        res_dict, title=None,
                        scale_lr=1, step_lr=1,
                        scale_hr=1, step_hr=1):

    xlr = res_dict['LR']['x']
    ulr = res_dict['LR']['u']
    L   = res_dict['LR']['size']

    xhr = res_dict['HR']['x']
    uhr = res_dict['HR']['u']
    H   = res_dict['HR']['size']

    return plot_velocity_field(xlr, ulr, L, xhr, uhr, H, 
                        title=title,
                        scale_lr=scale_lr, step_lr=step_lr,
                        scale_hr=scale_hr, step_hr=step_hr)
