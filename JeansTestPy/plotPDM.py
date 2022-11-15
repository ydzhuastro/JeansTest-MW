import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.tri as tri
from matplotlib.colors import LogNorm

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)

df = pd.read_csv("PotentialSolver/dspdm-W21.txt", sep="\s+", header=0, names=["R", "z", "pdm", "cdm", "gdm"], index_col=False)
# print(df)

pdm = df["pdm"]
cdm = df["cdm"]
gdm = df["gdm"]
# pdm[pdm<0.01] = 0.01
# cdm[cdm<0.01] = 0.01
# gdm[gdm<0.01] = 0.01
pdm[pdm>6] = 6
cdm[cdm>6] = 6
gdm[gdm>6] = 6
x = df["R"]
y = df["z"]
# zs = [np.log10(pdm), np.log10(cdm), np.log10(gdm)]
zs = [pdm, cdm, gdm]
# z = df["pdm"]

npts = len(x)
ngridx = 100
ngridy = 100

cntr1 = 0

fig, axs = plt.subplots(1,3, figsize=(8,3.1), sharey=True, sharex=True)
# fig, axs = plt.subplots(1,2, figsize=(6,3.1), sharey=True, sharex=True)

for (ax1, z) in zip(axs, zs):

    ax1.minorticks_on()
    ax1.tick_params(direction="in", which="both")
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.set(xlim=(-20, 20), ylim=(-20, 20))
    ax1.set_aspect('equal')

    # Create grid values first.
    xi = np.linspace(-30, 30, ngridx)
    yi = np.linspace(-30, 30, ngridy)

    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    #from scipy.interpolate import griddata
    #zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')

    # ax1.contour(xi, yi, zi.T, levels=15, linewidths=0.5, colors='k')
    if cntr1 == 0:
        cntr1 = ax1.contourf(xi, yi, zi, levels=50, cmap="hot", vmin=0, vmax=6)
    else:
        ax1.contourf(xi, yi, zi, levels=50, cmap="hot", vmin=0, vmax=6)

    ax1.tick_params(axis='both', colors='white', labelcolor='k')
    # fig.colorbar(cntr1, ax=ax1, label=r"$\log_{10}\Delta \Phi~({G M_\odot \rm kpc^{-1}})$")
    # ax1.plot(x, y, 'ko', ms=3)
    
    # ax1.set_xlabel(r"$R$ (kpc)")
    # ax1.set_ylabel(r"$z$ (kpc)")
# fig.subplots_adjust(hspace=0, wspace=0)
fig.colorbar(cntr1, ax = axs.ravel().tolist(), label=r"$\rho_{\rm DM}~({\rm GeV\,cm^{-3}})$", fraction=0.015, pad=0.01)
# fig.tight_layout(rect=(0.02, 0., 0.8, 1))
fig.text(0.5, 0.07, r"$R$ (kpc)", ha='center')
fig.text(0.07, 0.5, r"$z$ (kpc)", va='center', rotation='vertical')

plt.savefig("PDM-W22.pdf")
plt.show()


# # -----------------------
# # Interpolation on a grid
# # -----------------------
# # A contour plot of irregularly spaced data coordinates
# # via interpolation on a grid.

# ax1.minorticks_on()
# ax1.tick_params(direction="in", which="both")
# ax1.xaxis.set_ticks_position('both')
# ax1.yaxis.set_ticks_position('both')

# # Create grid values first.
# xi = np.linspace(-30, 30, ngridx)
# yi = np.linspace(-30, 30, ngridy)

# # Perform linear interpolation of the data (x,y)
# # on a grid defined by (xi,yi)
# triang = tri.Triangulation(x, y)
# interpolator = tri.LinearTriInterpolator(triang, z)
# Xi, Yi = np.meshgrid(xi, yi)
# zi = interpolator(Xi, Yi)

# # Note that scipy.interpolate provides means to interpolate data on a grid
# # as well. The following would be an alternative to the four lines above:
# #from scipy.interpolate import griddata
# #zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')


# # ax1.contour(xi, yi, zi.T, levels=15, linewidths=0.5, colors='k')
# cntr1 = ax1.contourf(xi, yi, zi.T, levels=15, cmap="RdBu_r")

# fig.colorbar(cntr1, ax=ax1, label=r"$\log_{10}\rho_{\rm PDM}~({\rm GeV\,cm^{-3}})$")
# # ax1.plot(x, y, 'ko', ms=3)
# ax1.set(xlim=(-20, 20), ylim=(-20, 20))
# ax1.set_xlabel(r"$R$ (kpc)")
# ax1.set_ylabel(r"$z$ (kpc)")


# plt.tight_layout()
# plt.savefig("PDM-W21.pdf")
# plt.show()