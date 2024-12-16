"""
Defines plotting functions for 2D FBPINN / PINN problems

This module is used by plot_trainer.py (and subsequently trainers.py)
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from cmap import Colormap

from matplotlib.gridspec import GridSpec

from fbpinns.plot_trainer_1D import _plot_setup, _to_numpy

import scienceplots
import matplotlib.tri as mtri
import triangle as tr
import time

# def circle(N, R):
#     i = np.arange(N)
#     theta = i * 2 * np.pi / N
#     pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
#     seg = np.stack([i, i + 1], axis=1) % N
#     return pts, seg

# pts0, seg0 = circle(60, 1.0)
# pts1, seg1 = circle(30, 0.5)
# pts = np.vstack([pts0, pts1])
# seg = np.vstack([seg0, seg1 + seg0.shape[0]])
# A = dict(vertices=pts, segments=seg, holes=[[0, 0]])
# T = tr.triangulate(A, "Fa0.0005") #note that the origin uses 'qpa0.05' here
# tri = mtri.Triangulation(T["vertices"][:,0], T["vertices"][:, 1], triangles=T["triangles"])
# xbatch_test_triangles = np.vstack([T["vertices"][:,0], T["vertices"][:, 1]]).T


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

    # plt.style.use(["science"])

    # SQUARE
    npzfile = dict(np.load("data/" + "heart" + "_mtri.npz", allow_pickle=True))["arr_0"]
    mtri = npzfile.item()

    fig, ax = plt.subplots(figsize=(0.6 * 8, 0.4 * 8))
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_aspect("equal")

    u_test = u_test.reshape(-1)
    np.save("heart" + f"_testNN[4, 8;2.6]e{i}_2", u_test)

    im = ax.tripcolor(mtri, u_test,
                      cmap=Colormap("crameri:batlow").to_mpl(), clim=(np.min(u_test), np.max(u_test)))


    cbar = fig.colorbar(im, orientation='vertical')
    plt.show()
    fig.savefig("heart" + f"_testNN[4, 8, 16;2.6]e{i}_2.png")
    # time.sleep(5)

    # xlim, ulim = _plot_setup(x_batch_test, u_exact)
    # xlim0 = x_batch_test.min(0), x_batch_test.max(0)

    # np.save(f"donut_epoch{i}_0", u_test)

    # f = plt.figure(figsize=(14,6))

    # plt.subplot(1,2,1)
    # plt.title(f"[{i}] Domain decomposition")
    # plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    # decomposition.plot(all_params, active=active, create_fig=False)
    # plt.xlim(xlim[0][0], xlim[1][0])
    # plt.ylim(xlim[0][1], xlim[1][1])
    # plt.gca().set_aspect("equal")
    
    
    # u1 = u_test[:, 0:1]
    # u2 = u_test[:, 1:2]
    
    # xlim, ulim = _plot_setup(x_batch_test, u_exact)
    # xlim0 = x_batch_test.min(0), x_batch_test.max(0)

    # f = plt.figure(figsize=(8,10))

    # # plot domain + x_batch
    # plt.subplot(2,2,1)
    # plt.title(f"[{i}] Domain decomposition")
    # plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    # decomposition.plot(all_params, active=active, create_fig=False)
    # plt.xlim(xlim[0][0], xlim[1][0])
    # plt.ylim(xlim[0][1], xlim[1][1])
    # plt.gca().set_aspect("equal")

    # plot full solutions
    # plt.subplot(3,2,2)
    # plt.title(f"[{i}] Difference")
    # _plot_test_im(u_exact - u_test, xlim0, ulim, n_test)

    # M = np.max(np.abs(u_test))

    # plt.subplot(2,2,3)
    # plt.title(f"[{i}] Full solution")
    # plt.pcolormesh(u_test.reshape(n_test).T, vmin=-M, vmax=M)
    # plt.colorbar()
    # plt.gca().set_aspect("equal")

    # plt.subplot(2,2,4)
    # plt.title(f"[{i}] Full solution")
    # _plot_test_im(u2, xlim0, ulim, n_test)
    # plt.gca().set_aspect("equal")

    # plt.subplot(3,2,4)
    # plt.title(f"[{i}] Ground truth")
    # _plot_test_im(u_exact, xlim0, ulim, n_test)

    # plot raw hist
    # plt.subplot(3,2,5)
    # plt.title(f"[{i}] Raw solutions")
    # plt.hist(us_raw_test.flatten(), bins=100, label=f"{us_raw_test.min():.1f}, {us_raw_test.max():.1f}")
    # plt.legend(loc=1)
    # plt.xlim(-5,5)
    
    # plt.tight_layout()

    return (("test",fig),)

# @_to_numpy
# def plot_2D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

#     u = u_test[:,0:1]

#     fig, ax = plt.subplots(1, 1, figsize=(9, 6))

#     ax.set_title(f"Epoch [{i}] Full solution.")
    
#     N = 100
    
#     U = u.reshape((N, N))
    
#     X, Y = x_batch_test[:, 0].reshape((N, N)), x_batch_test[:, 1].reshape((N, N))
    
#     p = ax.pcolormesh(X, Y, U)
#     plt.colorbar(p)

#     plt.tight_layout()
#     fig.savefig("tmp_plot")

#     return (("test", fig),)

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







