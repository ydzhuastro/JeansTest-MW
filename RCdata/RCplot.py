#!/usr/bin/env python

from matplotlib.pyplot import legend
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',
    # "font.size": 20,
    "mathtext.fontset":'stix',
}
rcParams.update(config)

legend_config = {
    "family":'Times New Roman',
    "size": 9
}

# [0]R, [1]Newtonian, [2]Newtonian+CDM, [3]QUMOND
model= np.loadtxt("PotentialSolver/dsvel-W21.txt")

# [0]R, [1]V, [2]Verr(+,-)
FileList = ("RCdata/Eilers2019.dat",
            "RCdata/Mroz2019.dat",
            "RCdata/Chrobakova2020_z=0.dat",
            "PotentialSolver/RC_stddev_W21.txt")

DataList = ("Eilers et al. (2019)",
            "Mróz et al. (2019)",
            "Chrobáková et al. (2020)",
            "Averaged")

f = []
for filename in FileList:
    f.append(np.loadtxt(filename))
f = np.array(f)

#color sets of lines: [Baryonic, Baryonic+CDM, Baryonic+QUMOND, Baryonic+MOG]
linecolor = ["C1", "C0", "green", "C3"]


plt.figure(figsize=(8.0, 4.0))
ax = plt.subplot(111)

plt.errorbar(x=f[0][:,0], y=f[0][:,1], yerr=f[0][:,2:4].T, fmt='bd', ms=1, alpha=0.3, elinewidth=0.5, capsize=1, capthick=0.5, label=DataList[0])
plt.errorbar(x=f[1][:,0], y=f[1][:,1], yerr=f[1][:,2], xerr=f[1][:,3], fmt='o', c='gray', alpha=0.2, ms=0.7, elinewidth=0.5, capsize=1, capthick=0.5, label=DataList[1])
plt.errorbar(x=f[2][:,0], y=f[2][:,1], yerr=f[2][:,2], fmt='mp', ms=1, alpha=0.3, elinewidth=0.5, capsize=1, capthick=0.5, label=DataList[2])
plt.errorbar(x=f[3][:,1], y=f[3][:,2], yerr=f[3][:,3], fmt='cs', c='C9', ms=2, elinewidth=0.8, capsize=1.5, capthick=0.8, label="Averaged")

plt.plot(model[:,0], model[:,1], linewidth=1, c=linecolor[0], label="Newtonian")
plt.plot(model[:,0], model[:,3], linewidth=1, c=linecolor[2], label="QUMOND")
plt.plot(model[:,0], model[:,2], linewidth=1, c=linecolor[1], label="Newtonian + DM")
plt.plot(model[:,0], model[:,4], linewidth=1, c=linecolor[3], label="MOG")

plt.xlabel("$R\, [\mathrm{kpc}]$")
plt.ylabel("$v_\mathrm{c}\, [\mathrm{km/s}]$")
plt.xlim(3, 30)
plt.ylim(50, 400)
plt.legend(prop=legend_config)
# plt.grid(linestyle='-.')
ax.minorticks_on()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(direction="in", which="both")

plt.savefig("RC_W22_stddev.pdf", dpi=300)