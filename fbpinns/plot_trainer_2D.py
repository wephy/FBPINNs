"""
Defines plotting functions for 2D FBPINN / PINN problems

This module is used by plot_trainer.py (and subsequently trainers.py)
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from fbpinns.plot_trainer_1D import _plot_setup, _to_numpy

def _plot_test_im(u_test, xlim, ulim, n_test, it=None, bounds=True):
    u_test = u_test.reshape(n_test)
    if it is not None:
        u_test = u_test[:,:,it]# for 3D
    # if bounds:
    #     plt.imshow(u_test.T,# transpose as jnp.meshgrid uses indexing="ij"
    #             origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
    #             cmap="viridis", vmin=ulim[0], vmax=ulim[1])
    # else:
    plt.pcolormesh(u_test.T),# transpose as jnp.meshgrid uses indexing="ij"
                # origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
                # cmap="viridis")
    plt.colorbar()
    # plt.xlim(xlim[0][0], xlim[1][0])
    # plt.ylim(xlim[0][1], xlim[1][1])
    # plt.gca().set_aspect("equal")

@_to_numpy
def plot_2D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    u_test = u_test[:,0:1]

    print(u_test.shape)

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch_test.min(0), x_batch_test.max(0)

    fig = plt.figure(figsize=(14,14))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 2], hspace=0.3, wspace=0.3)

    # # plot domain + x_batch
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax1.set_title(f"[{i}] Domain decomposition")
    ax1.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    decomposition.plot(all_params, active=active, create_fig=False)
    ax1.set_xlim(xlim[0][0]-1, xlim[1][0]+1)
    ax1.set_ylim(xlim[0][1]-1, xlim[1][1]+1)
    
    ax2 = fig.add_subplot(gs[1, 0:3])
    ax2.set_title(f"Epoch [{i}] Full solution")
    
    xs, ts = x_batch_test[:, 0], x_batch_test[:, 1]
    X, T = np.meshgrid(xs, ts)
    ax2.pcolormesh(X, T, u_test.reshape(-1))
    
    
    # ax1.set_aspect("equal")

    # plot full solutions
    # diff = (u_exact - u_test)
    # ax2 = fig.add_subplot(gs[1, 0])
    # ax2.set_title(f"[{i}] Difference")
    # _plot_test_im(diff, xlim0, [-0.01, 0.01], n_test, bounds=False)


    # _plot_test_im(u_test, xlim0, ulim, n_test, bounds=False)

    # ax4 = fig.add_subplot(gs[1, 2])
    # ax4.set_title(f"[{i}] Ground truth")
    # _plot_test_im(u_exact, xlim0, ulim, n_test, bounds=False)

    # plot raw hist
    # ax5 = fig.add_subplot(gs[0, 2])
    # ax5.set_title(f"[{i}] Difference hist")
    # diff = diff.flatten()
    # emin = diff.min()
    # emax = diff.max()
    # _, bins = np.histogram(diff, bins=100)
    # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    # ax5.hist(diff.flatten(), bins=logbins)#, label=f"{emin:.1f}, {emax:.1f}")
    # # plt.xscale('log')
    
    plt.tight_layout()

    return (("test", fig),)

@_to_numpy
def plot_2D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch.min(0), x_batch.max(0)

    f = plt.figure(figsize=(8,10))

    # plot x_batch
    plt.subplot(3,2,1)
    plt.title(f"[{i}] Training points")
    plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")

    # plot full solution
    plt.subplot(3,2,2)
    plt.title(f"[{i}] Difference")
    _plot_test_im(u_exact - u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,3)
    plt.title(f"[{i}] Full solution")
    _plot_test_im(u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,4)
    plt.title(f"[{i}] Ground truth")
    _plot_test_im(u_exact, xlim0, ulim, n_test)

    # plot raw hist
    plt.subplot(3,2,5)
    plt.title(f"[{i}] Raw solution")
    plt.hist(u_raw_test.flatten(), bins=100, label=f"{u_raw_test.min():.1f}, {u_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)







