#!/Users/mahaixia/opt/anaconda3/envs/jeansbase/bin/python
"""
    Perform Jeans Tests on gravitational potential models of the MW.
    (c) 2022 Yongda Zhu [yzhu144@ucr.edu]; Haixia Ma; 

    - Apr 23, 2022; In this version, we only compare the observational term to the predicted rho*d_phi/d_R or rho*d_phi/d_z;
"""
import warnings
import obs_gaia2
import obs_binney2014_rc
# from GalPot import GalaxyPotential
import globalvars
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from functools import lru_cache
import pandas as pd
import scipy.interpolate as interp
from scipy.interpolate import interp2d
from scipy import stats
# plt.style.use("ggplot")

from matplotlib import rc
# rc('font',**{'family':'Times New Roman'})
rc('text', usetex=True)
font = {'family' : 'Times New Roman',
        'size'   : 11}
rc('font', **font)


warnings.simplefilter('error', UserWarning)

BASE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"
DS_OUTPUT_DIR = BASE_DIR + "../"
print(BASE_DIR, "\n", DS_OUTPUT_DIR)

KPCpMYRtKMpS = 978.4620750887875

## test
# print(Galaxy.Potential(8, 0))
# example = StellarDisk("/Users/ydzhu/workspace/JeansTestPy/pot/CDMMCR0.Tpot")
## test


h = 0.005

class DSProfileM17(object):
    def __init__(self):
        # read poi-2022.edp outputs
        # result = "/Users/mahaixia/Git_Workspace/phantom2/dsmatrix-GAIA-M17.ds"
        result = "/Users/ydzhu/workspace/phantom-2.2/dsmatrix-GAIA-M17.ds"
        with open(result, 'rb') as f:
            N = np.fromfile(f, dtype=np.intc, count=1)[0]
            dsR = np.fromfile(f, np.double, N)
            dsz = np.fromfile(f, np.double, N)
            dsQumondPhi = np.fromfile(f, np.double, N)
            dsColddmPhi = np.fromfile(f, np.double, N)
            dsNewtonPhi = np.fromfile(f, np.double, N)
            dsMoffatPhi = np.fromfile(f, np.double, N)
            dsrho = np.fromfile(f, np.double, N)
        self.interp_QumondPhi = lambda x, y: interp.griddata(np.array([dsR, dsz]).T, dsQumondPhi, np.array([x, y]).T, method="cubic")[0]
        self.interp_ColddmPhi = lambda x, y: interp.griddata(np.array([dsR, dsz]).T, dsColddmPhi, np.array([x, y]).T, method="cubic")[0]
        self.interp_NewtonPhi = lambda x, y: interp.griddata(np.array([dsR, dsz]).T, dsNewtonPhi, np.array([x, y]).T, method="cubic")[0]
        self.interp_MoffatPhi = lambda x, y: interp.griddata(np.array([dsR, dsz]).T, dsMoffatPhi, np.array([x, y]).T, method="cubic")[0]
        self.interp_rho = lambda x, y: interp.griddata(np.array([dsR, dsz]).T, dsrho, np.array([x, y]).T, method="cubic")[0]

    # add thin disk of M17
    def thin_disk_density(self, R, z):
        Sigma0thin = 952e6
        Rdthin = 2.40037
        zdthin = 0.3

        return 0.5*Sigma0thin/zdthin*np.exp(-np.abs(z)/zdthin - R/Rdthin)

    def thick_disk_density(self, R, z):
        Sigma0thick = 119e6
        Rdthick = 3.47151
        zdthick = 0.9

        return 0.5*Sigma0thick/zdthick*np.exp(-np.abs(z)/zdthick - R/Rdthick)

    def total_disk_density(self, R, z):
        return 0.85*self.thin_disk_density(R, z) + 0.15*self.thick_disk_density(R, z) # no gas!

    def total_density(self, R, z):
        # fine
        return self.interp_rho(R, z)

    def Potential_derivative_Newton_R(self, R, z):
        return (self.interp_NewtonPhi(R + h, z) - self.interp_NewtonPhi(R - h, z)) / (2*h)
    def Potential_derivative_Newton_z(self, R, z):
        return (self.interp_NewtonPhi(R, z + h) - self.interp_NewtonPhi(R, z - h)) / (2*h)

    def Potential_derivative_QUMOND_R(self, R, z):
        return (self.interp_QumondPhi(R + h, z) - self.interp_QumondPhi(R - h, z)) / (2*h)
    def Potential_derivative_QUMOND_z(self, R, z):
        return (self.interp_QumondPhi(R, z + h) - self.interp_QumondPhi(R, z - h)) / (2*h)

    def Potential_derivative_CDM_R(self, R, z):
        return (self.interp_ColddmPhi(R + h, z) - self.interp_ColddmPhi(R - h, z)) / (2*h)
    def Potential_derivative_CDM_z(self, R, z):
        return (self.interp_ColddmPhi(R, z + h) - self.interp_ColddmPhi(R, z - h)) / (2*h)

    def Potential_derivative_MOG_R(self, R, z):
        return (self.interp_MoffatPhi(R + h, z) - self.interp_MoffatPhi(R - h, z)) / (2*h)
    def Potential_derivative_MOG_z(self, R, z):
        return (self.interp_MoffatPhi(R, z + h) - self.interp_MoffatPhi(R, z - h)) / (2*h)


def MOND_v(y):
    return 0.5 * np.sqrt(1.0 + 4.0 / y) + 0.5


def _JeansTests(R: float, z: float, galaxy, tracer_density, sigmaR, sigmaz, sigmaT, VT, alpha):
    """
    Perform Jeans Tests: core functions

    Args:
        R, z: the range you wanna cover
        galaxy class
        tracer_density: density profile of the tracer
        sigmaR, dsigmaR: function of radial velocity dispersion func(R, z) from observation
        sigmaz, dsigmaz
        sigmaT, dsigmaT: theta
        vT, dvT
        alpha: tilt angle of the velocity ellipsoid as a function of (R, z)
    
    Returns:
        TR, Tz
    """

    # OBSERVATION
    ## firstly, calculate kz, ktheta, kappa, xi
    def kz(local_R, local_z): return (
        sigmaz(local_R, local_z)/sigmaR(local_R, local_z))**2
    def kT(local_R, local_z): return (
        sigmaT(local_R, local_z)/sigmaR(local_R, local_z))**2
    def kappa(local_R, local_z): return 0.5 * \
        np.tan(2*alpha(local_R, local_z))*(1-kz(local_R, local_z))

    def xi(local_R, local_z): return kappa(
        local_R, local_z) / kz(local_R, local_z)

    ## combined quantities
    def RhosigmaR(local_R, local_z):
        return tracer_density(local_R, local_z)*sigmaR(local_R, local_z)*sigmaR(local_R, local_z)

    def Rhosigmaz(local_R, local_z):
        return tracer_density(local_R, local_z)*sigmaz(local_R, local_z)*sigmaz(local_R, local_z)

    ## calculate derivatives
    h = 0.005      # step
    RhosigmaRR = (RhosigmaR(R + h, z) - RhosigmaR(R - h, z))/(2*h)
    RhosigmaRz = (RhosigmaR(R, z + h) - RhosigmaR(R, z - h))/(2*h)
    RhosigmazR = (Rhosigmaz(R + h, z) - Rhosigmaz(R - h, z))/(2*h)
    Rhosigmazz = (Rhosigmaz(R, z + h) - Rhosigmaz(R, z - h))/(2*h)

    kappaz = (kappa(R, z + h) - kappa(R, z - h)) / (2*h)
    xiR = (xi(R + h, z) - xi(R - h, z)) / (2*h)

    Rho = tracer_density(R, z)
    ## calculate TR
    # TRn = - Rho * PR
    TRd = RhosigmaRR + ((1-kT(R, z))/R + kappaz)*RhosigmaR(R, z) + \
        kappa(R, z)*RhosigmaRz - Rho * VT(R, z) * VT(R, z) / R
    # TRd0 = RhosigmaRR
    # TRd1 = ((1-kT(R, z))/R + kappaz)*RhosigmaR(R, z)
    # TRd2 = kappa(R, z)*RhosigmaRz
    # TRd3 = Rho * VT(R, z) * VT(R, z) / R
    # TR = (TRn-TRd)/TRd
    TR = -TRd/Rho

    ## calculate Tz
    Tzd = Rhosigmazz + (xi(R, z)/R + xiR)*Rhosigmaz(R, z) + xi(R, z)*RhosigmazR
    # Tzd0 = Rhosigmazz
    # Tzd1 = (xi(R, z)/R + xiR)*Rhosigmaz(R, z)
    # Tzd2 = xi(R, z)*RhosigmazR
    Tz = -Tzd/Rho

    return TR, Tz

def JeansTests(R: float, z: float, galaxy, tracer_density, sigmaR, dsigmaR, sigmaz, dsigmaz, sigmaT, dsigmaT, VT, dVT, alpha):
    """
    Perform Jeans Tests

    Args:
        R, z: the range you wanna cover
        galaxy class
        tracer_density: density profile of the tracer
        sigmaR, dsigmaR: function of radial velocity dispersion func(R, z) from observation
        sigmaz, dsigmaz
        sigmaT, dsigmaT: theta
        vT, dvT
        alpha: tilt angle of the velocity ellipsoid as a function of (R, z)
    
    Returns:
        TR, dTR, Tz, dTz: float
    """
    TR, Tz = _JeansTests(R, z, galaxy, tracer_density, sigmaR, sigmaz, sigmaT, VT, alpha)

    # calculate derivatives
    h = 0.005

    def dds(f):
        return lambda *args: f(*args) + h
    
    def dfb(f):
        return lambda *args: f(*args) - h

    # calculate differentials
    _a0, _b0 = _JeansTests(R, z, galaxy, tracer_density, dds(sigmaR), sigmaz, sigmaT, VT, alpha)
    _a1, _b1 = _JeansTests(R, z, galaxy, tracer_density, dfb(sigmaR), sigmaz, sigmaT, VT, alpha)
    dTR_dsigmaR = (_a0 - _a1) / (2*h)
    dTz_dsigmaR = (_b0 - _b1) / (2*h)#

    _a0, _b0 = _JeansTests(R, z, galaxy, tracer_density, sigmaR, dds(sigmaz), sigmaT, VT, alpha)
    _a1, _b1 = _JeansTests(R, z, galaxy, tracer_density, sigmaR, dfb(sigmaz), sigmaT, VT, alpha)
    dTR_dsigmaz = (_a0 - _a1) / (2*h)
    dTz_dsigmaz = (_b0 - _b1) / (2*h)

    _a0, _b0 = _JeansTests(R, z, galaxy, tracer_density, sigmaR, sigmaz, dds(sigmaT), VT, alpha)
    _a1, _b1 = _JeansTests(R, z, galaxy, tracer_density, sigmaR, sigmaz, dfb(sigmaT), VT, alpha)
    dTR_dsigmaT = (_a0 - _a1) / (2*h)
    dTz_dsigmaT = (_b0 - _b1) / (2*h)

    _a0, _b0 = _JeansTests(R, z, galaxy, tracer_density, sigmaR, sigmaz, sigmaT, dds(VT), alpha)
    _a1, _b1 = _JeansTests(R, z, galaxy, tracer_density, sigmaR, sigmaz, sigmaT, dfb(VT), alpha)
    dTR_VT = (_a0 - _a1) / (2*h)
    dTz_VT = (_b0 - _b1) / (2*h)

    _a0, _b0 = _JeansTests(R, z, galaxy, dds(tracer_density), sigmaR, sigmaz, sigmaT, VT, alpha)
    _a1, _b1 = _JeansTests(R, z, galaxy, dfb(tracer_density), sigmaR, sigmaz, sigmaT, VT, alpha)
    dTR_tracer = (_a0 - _a1) / (2*h)
    dTz_tracer = (_b0 - _b1) / (2*h)

    dsR = dsigmaR(R, z)
    dsz = dsigmaz(R, z)
    dsT = dsigmaT(R, z)
    dvT = dVT(R, z)
    dtracer = tracer_density(R, z)*0.2 # plus a nominal error for tracer to be 20%

    dTR = np.sqrt(dTR_dsigmaR**2*dsR**2 + dTR_dsigmaz**2*dsz**2 + dTR_dsigmaT**2*dsT**2 + dTR_VT**2*dvT**2 + dTR_tracer**2*dtracer**2)
    dTz = np.sqrt(dTz_dsigmaR**2*dsR**2 + dTz_dsigmaz**2*dsz**2 + dTz_dsigmaT**2*dsT**2 + dTz_VT**2*dvT**2 + dTz_tracer**2*dtracer**2)

    return TR, Tz, dTR, dTz

def rotation_curve(Rlist: np.ndarray, galaxy, mode=None):
    PR = np.zeros(len(Rlist))
    for i in range(len(Rlist)):
        _, PR[i], _ = galaxy.Potential_derivatives(Rlist[i], 0)
    PR = np.abs(PR)
    V = np.sqrt(PR*Rlist)*globalvars.KPCpMYRtKMpS
    MPR = np.array([x*MOND_v(x/globalvars.a0) for x in PR])
    MV = np.sqrt(MPR*Rlist)*globalvars.KPCpMYRtKMpS
    if mode == "MOND":
        return MV
    else:
        return V

def _1d_data(obs, z, Rlist, model="M17", density_type="thin profile"):
    if model == "M17":
        galaxy = DSProfileM17()
    else:
        print("ERROR: model unavailable!", model)
        exit()

    TR = np.zeros(len(Rlist))
    Tz = np.zeros(len(Rlist))
    dTR = np.zeros(len(Rlist))
    dTz = np.zeros(len(Rlist))

    NTR = np.zeros(len(Rlist))
    DTR = np.zeros(len(Rlist))
    MTR = np.zeros(len(Rlist))
    GTR = np.zeros(len(Rlist))
    NTz = np.zeros(len(Rlist))
    DTz = np.zeros(len(Rlist))
    MTz = np.zeros(len(Rlist))
    GTz = np.zeros(len(Rlist))

    if density_type == "thin profile":
        tracer_profile = galaxy.thin_disk_density
    elif density_type == "thick profile":
        tracer_profile = galaxy.thick_disk_density
    elif density_type == "thin + thick profile":
        tracer_profile = galaxy.total_disk_density
    elif density_type == "total":
        tracer_profile = galaxy.total_density
    else:
        print("ERROR: tracer type unavailable!", density_type)
        exit()

    if obs == "gaia2":

        for i in range(len(Rlist)):
            TR[i], Tz[i], dTR[i], dTz[i] = JeansTests(Rlist[i], z, galaxy, tracer_profile, obs_gaia2.sigmaR, obs_gaia2.EsigmaR,
                                                obs_gaia2.sigmaz, obs_gaia2.Esigmaz, obs_gaia2.sigmaT, obs_gaia2.EsigmaT, obs_gaia2.VT, obs_gaia2.EVT, obs_gaia2.alpha)
            TR[i]*=KPCpMYRtKMpS*KPCpMYRtKMpS
            Tz[i]*=KPCpMYRtKMpS*KPCpMYRtKMpS
            dTR[i]*=KPCpMYRtKMpS*KPCpMYRtKMpS
            dTz[i]*=KPCpMYRtKMpS*KPCpMYRtKMpS

    elif obs == "binney2014":

        for i in range(len(Rlist)):
            TR[i], Tz[i], dTR[i], dTz[i] = JeansTests(Rlist[i], z, galaxy, tracer_profile, obs_binney2014_rc.sigmaR, obs_binney2014_rc.EsigmaR,
                                                obs_binney2014_rc.sigmaz, obs_binney2014_rc.Esigmaz, obs_binney2014_rc.sigmaT, obs_binney2014_rc.EsigmaT, obs_binney2014_rc.VT, obs_binney2014_rc.EVT, obs_binney2014_rc.alpha)
            TR[i]*=KPCpMYRtKMpS*KPCpMYRtKMpS
            Tz[i]*=KPCpMYRtKMpS*KPCpMYRtKMpS
            dTR[i]*=KPCpMYRtKMpS*KPCpMYRtKMpS
            dTz[i]*=KPCpMYRtKMpS*KPCpMYRtKMpS
    
    for i in range(len(Rlist)):
        # NTR[i] = tracer_profile(Rlist[i], z)*galaxy.Potential_derivative_Newton_R(Rlist[i], z)
        # NTz[i] = tracer_profile(Rlist[i], z)*galaxy.Potential_derivative_Newton_z(Rlist[i], z)
        # DTR[i] = tracer_profile(Rlist[i], z)*galaxy.Potential_derivative_CDM_R(Rlist[i], z)
        # DTz[i] = tracer_profile(Rlist[i], z)*galaxy.Potential_derivative_CDM_z(Rlist[i], z)
        # MTR[i] = tracer_profile(Rlist[i], z)*galaxy.Potential_derivative_QUMOND_R(Rlist[i], z)
        # MTz[i] = tracer_profile(Rlist[i], z)*galaxy.Potential_derivative_QUMOND_z(Rlist[i], z)
        # GTR[i] = tracer_profile(Rlist[i], z)*galaxy.Potential_derivative_MOG_R(Rlist[i], z)
        # GTz[i] = tracer_profile(Rlist[i], z)*galaxy.Potential_derivative_MOG_z(Rlist[i], z)

        NTR[i] = galaxy.Potential_derivative_Newton_R(Rlist[i], z)*KPCpMYRtKMpS*KPCpMYRtKMpS
        NTz[i] = galaxy.Potential_derivative_Newton_z(Rlist[i], z)*KPCpMYRtKMpS*KPCpMYRtKMpS
        DTR[i] = galaxy.Potential_derivative_CDM_R(Rlist[i], z)*KPCpMYRtKMpS*KPCpMYRtKMpS
        DTz[i] = galaxy.Potential_derivative_CDM_z(Rlist[i], z)*KPCpMYRtKMpS*KPCpMYRtKMpS
        MTR[i] = galaxy.Potential_derivative_QUMOND_R(Rlist[i], z)*KPCpMYRtKMpS*KPCpMYRtKMpS
        MTz[i] = galaxy.Potential_derivative_QUMOND_z(Rlist[i], z)*KPCpMYRtKMpS*KPCpMYRtKMpS
        GTR[i] = galaxy.Potential_derivative_MOG_R(Rlist[i], z)*KPCpMYRtKMpS*KPCpMYRtKMpS
        GTz[i] = galaxy.Potential_derivative_MOG_z(Rlist[i], z)*KPCpMYRtKMpS*KPCpMYRtKMpS
    
    # observation; models
    return np.array(TR), np.array(Tz), np.array(dTR), np.array(dTz), np.array(NTR), np.array(DTR), np.array(MTR), np.array(GTR), np.array(NTz), np.array(DTz), np.array(MTz), np.array(GTz)

def chisqr(obs, exp, error):
    chisqr = 0
    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i]-exp[i])**2)/(error[i]**2)
        # chisqr = chisqr + ((obs[i]-exp[i])**2)/(0.1**2)
    p = 1 - stats.chi2.cdf(chisqr, len(obs)-1)
    return chisqr/(len(obs)-1), p

if __name__ == "__main__":
    # plot
    zlist = [1.18, 0.75, 0.44]
    Rlist = np.arange(6, 12, 0.5)

    binney_rlist = np.array([[7.52, 8.37], [7.48, 8.41], [7.51, 8.36], [7.61, 8.36]])

    #color sets of lines: [Baryonic, Baryonic+DM, Baryonic+QUMOND, Baryonic+MOG]
    linecolor = ["C1", "C0", "green", "C3"]

    ds = "mid"
    if True:
        # Figure TR, Tz
        figR, axrcR = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(13, 4))
        figz, axrcz = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(13, 4))
        for i in range(len(axrcR)):
                axrcR[i].minorticks_on()
                axrcR[i].tick_params(direction="in", which="both")
                axrcR[i].xaxis.set_ticks_position('both')
                axrcR[i].yaxis.set_ticks_position('both')
                # axrcR[i][j].set_ylim(-0.62, 0.5)
                # axrcR[i][j].set_ylim(1e-2, 1e6)
                # axrcR[i][j].set_yscale("log")
                axrcz[i].minorticks_on()
                axrcz[i].tick_params(direction="in", which="both")
                axrcz[i].xaxis.set_ticks_position('both')
                axrcz[i].yaxis.set_ticks_position('both')
                # axrcz[i][j].set_ylim(-3, 4.6)
                # axrcz[i][j].set_yscale("log")
                # axrcz[i][j].set_ylim(1e2, 1e5)

        array_TR = []
        array_Tz = []
        array_dTR = []
        array_dTz = []

        array_NTR  = []
        array_DTR  = []
        array_MTR  = []
        array_GTR  = [] #MOG
        array_NTz  = []
        array_DTz  = []
        array_MTz  = []
        array_GTz  = []

        zidx = 0
        for i in range(3):
                TR, Tz, dTR, dTz, NTR, DTR, MTR, GTR, NTz, DTz, MTz, GTz = _1d_data("gaia2", zlist[zidx], Rlist, model="M17", density_type="thin + thick profile")
                array_TR  .append(TR)
                array_Tz  .append(Tz)
                array_dTR .append(dTR)
                array_dTz .append(dTz)
                
                array_NTR .append(NTR )
                array_DTR .append(DTR )
                array_MTR .append(MTR )
                array_GTR .append(GTR )
                array_NTz .append(NTz )
                array_DTz .append(DTz )
                array_MTz .append(MTz )
                array_GTz .append(GTz )

                ds2 = "steps-mid"
                ds2 = None
                axrcR[i].text(0.15, 0.9, r"$z=%2.1f~{\rm kpc}$"%zlist[zidx], transform=axrcR[i].transAxes)
                line_R_obs, = axrcR[i].plot(Rlist,TR, "--", c="k", lw=1.0)
                shade_R_obs = axrcR[i].fill_between(Rlist, TR - dTR, TR + dTR, facecolor="sandybrown", alpha=0.4, label="Observation")
                axrcR[i].fill_between(Rlist, TR - 2*dTR, TR + 2*dTR, facecolor="sandybrown", alpha=0.2)
                line_R_net, = axrcR[i].plot(Rlist, NTR, c=linecolor[0], alpha=1, ds=ds2, lw=0.8, label="Newtonian", ls="-.")
                line_R_mog, = axrcR[i].plot(Rlist, GTR, c=linecolor[3], alpha=1, ds=ds2, label="MOG", ls="dotted")
                line_R_cdm, = axrcR[i].plot(Rlist, DTR, c=linecolor[1], alpha=1, ds=ds2, label="Newtonian + DM", ls=(0, (5, 1)))
                line_R_qmd, = axrcR[i].plot(Rlist, MTR, c=linecolor[2], alpha=1, ds=ds2, label="QUMOND")

                axrcz[i].text(0.15, 0.9, r"$z=%2.1f~{\rm kpc}$"%zlist[zidx], transform=axrcz[i].transAxes)
                line_z_obs, = axrcz[i].plot(Rlist,Tz, "--", c="k", lw=1.0)
                shade_z_obs = axrcz[i].fill_between(Rlist, Tz - dTz, Tz + dTz, facecolor="sandybrown", alpha=0.4, label="Observation")
                axrcz[i].fill_between(Rlist, Tz - 2*dTz, Tz + 2*dTz, facecolor="sandybrown", alpha=0.2)
                line_z_net, = axrcz[i].plot(Rlist, NTz, c=linecolor[0], alpha=1, ds=ds2, lw=0.8, label="Newtonian", ls="-.")
                line_z_mog, = axrcz[i].plot(Rlist, GTz, c=linecolor[3], alpha=1, ds=ds2, label="MOG", ls="dotted")
                line_z_cdm, = axrcz[i].plot(Rlist, DTz, c=linecolor[1], alpha=1, ds=ds2, label="Newtonian + DM", ls=(0, (5, 1)))
                line_z_qmd, = axrcz[i].plot(Rlist, MTz, c=linecolor[2], alpha=1, ds=ds2, label="QUMOND")
                # print(MTz, GTz)

                # binney
                TR, Tz, dTR, dTz, _, _, _, _, _, _, _, _ = _1d_data("binney2014", zlist[zidx], binney_rlist[zidx], model="M17", density_type="thin + thick profile")
                # print(TR, Tz, dTR, dTz)
                axrcR[i].errorbar(binney_rlist[zidx], TR, yerr=dTR, ls="none", capsize=2, elinewidth=0.6, c="k", marker="D", ms=2, zorder=10)
                axrcz[i].errorbar(binney_rlist[zidx], Tz, yerr=dTz, ls="none", capsize=2, elinewidth=0.6, c="k", marker="D", ms=2, zorder=10)
                # legend
                if i == 1:
                    axrcR[i].legend([(line_R_obs, shade_R_obs), line_R_qmd, line_R_cdm, line_R_mog, line_R_net], 
                                    ["Observation", "QUMOND", "Newtonian + DM", "MOG", "Newtonian"], frameon=False, fontsize=10, loc="upper right")
                    axrcz[i].legend([(line_z_obs, shade_z_obs), line_z_qmd, line_z_cdm, line_z_mog, line_z_net], 
                                    ["Observation", "QUMOND", "Newtonian + DM", "MOG", "Newtonian"], frameon=False, fontsize=10, loc="upper right")
                    # axrcR[i].plot([9.3, 9.7], [8000, 8000], "--", c="k", lw=1.0)
                    # axrcz[i].plot([9.3, 9.7], [4950, 4950], "--", c="k", lw=1.0)

                
                zidx = zidx + 1

        # axrcR[0].text(0.84, 0.9, "M17", transform=axrcR[0].transAxes, color="b")
        # axrcz[0].text(0.84, 0.9, "M17", transform=axrcz[0].transAxes, color="b")

        figR.tight_layout(rect=(0.04, 0.04, 0.97, 0.97))
        figR.subplots_adjust(hspace=0, wspace=0)
        figR.text(0.5, 0.03, r"$R~[\rm kpc]$", ha='center')
        figR.text(0.03, 0.5, r"$T_R~[\rm (km/s)^2/kpc]$", va='center', rotation='vertical')

        figz.tight_layout(rect=(0.04, 0.04, 0.97, 0.97))
        figz.subplots_adjust(hspace=0, wspace=0)
        figz.text(0.5, 0.03, r"$R~[\rm kpc]$", ha='center')
        figz.text(0.03, 0.5, r"$T_z~[\rm (km/s)^2/kpc]$", va='center', rotation='vertical')
        figR.savefig("TR-M17.pdf")
        figz.savefig("Tz-M17.pdf")
        # plt.show()

        plt.close()

        array_TR =np.concatenate(array_TR )
        array_Tz =np.concatenate(array_Tz )
        array_dTR =np.concatenate(array_dTR )
        array_dTz =np.concatenate(array_dTz )
        array_NTR =np.concatenate(array_NTR )
        array_DTR =np.concatenate(array_DTR )
        array_MTR =np.concatenate(array_MTR )
        array_GTR =np.concatenate(array_GTR )
        array_NTz =np.concatenate(array_NTz )
        array_DTz =np.concatenate(array_DTz )
        array_MTz =np.concatenate(array_MTz )
        array_GTz =np.concatenate(array_GTz )
        print("chi2 M17 TR", "Newton ", chisqr(array_NTR, array_TR, array_dTR))
        print("chi2 M17 Tz", "Newton ", chisqr(array_NTz, array_Tz, array_dTz))
        print("chi2 M17 TR", "Colddm ", chisqr(array_DTR, array_TR, array_dTR))
        print("chi2 M17 Tz", "Colddm ", chisqr(array_DTz, array_Tz, array_dTz))
        print("chi2 M17 TR", "QUMOND ", chisqr(array_MTR, array_TR, array_dTR))
        print("chi2 M17 Tz", "QUMOND ", chisqr(array_MTz, array_Tz, array_dTz))
        print("chi2 M17 TR", "MOG    ", chisqr(array_GTR, array_TR, array_dTR))
        print("chi2 M17 Tz", "MOG    ", chisqr(array_GTz, array_Tz, array_dTz))

    print("all set!")
    exit()