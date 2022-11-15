#!/usr/bin/env python
# -*- coding: utf-8 -*-

# plot potential dofference compared to the Newtonian case

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.tri as tri
from matplotlib.colors import LogNorm

import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin = 0.03, vmax = 0.12)

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)

kpc = 3.085678E19
MSun = 1.989E30
Myr = 3.1536E13
G = 4.493531540994E-12
KGpM3tGeVpCM3 = 5.6095888E20
toGeV = 3.797641792E-08
MpSStKPCpMYRMYR = 1.0/(kpc/Myr/Myr)
KMpStKPCpMYR = 1000.0/(kpc/Myr)
KPCpMYRtKMpS = kpc/Myr/1000.0
a0 = 1.2E-10*MpSStKPCpMYRMYR
Faclgt = 32


df = pd.read_csv("PotentialSolver/dspn-W21.txt", sep="\s+", header=0, names=["R", "z", "pn", "pc", "pp", "pg"], index_col=False)

pn = df["pn"]
pg = pn - df["pg"]
pp = pn - df["pp"]
pc = pn - df["pc"]

# pdm[pdm<0.01] = 0.01
x = df["R"]
y = df["z"]
zs = [pp, pc, pg]

npts = len(x)
ngridx = 100
ngridy = 100

cntr1 = 0

fig, axs = plt.subplots(1,3, figsize=(8,3.5), sharey=True, sharex=True)

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

    if cntr1 == 0:
        cntr1 = ax1.contourf(xi, yi, zi, levels=50, cmap="Greys", norm=norm)
    else:
        ax1.contourf(xi, yi, zi, levels=50, cmap="Greys", norm=norm)

fig.colorbar(cntr1, ax = axs.ravel().tolist(), label=r"$\Delta \Phi~({G M_\odot \rm kpc^{-1}})$", fraction=0.015, pad=0.03)
fig.text(0.5, 0.05, r"$R$ (kpc)", ha='center')
fig.text(0.05, 0.5, r"$z$ (kpc)", va='center', rotation='vertical')

plt.savefig("potential-difference-W21.pdf")
plt.show()