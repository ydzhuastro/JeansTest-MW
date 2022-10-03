#!/usr/bin/env python
"""
    Observations form GAIA DR2
"""

import numpy as np
from globalvars import KMpStKPCpMYR
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy import optimize
from functools import lru_cache

from uncertainties import ufloat
from uncertainties import umath 
from uncertainties import unumpy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plane_fitting import linear_2d_fit, quadratic_2d_fit

# in kpc/Myr
# read from Huang+2020
filename = "JeansTestPy/binned_results.npz"
ff = np.load(filename)
R_bin=ff["R_bin"]
z_bin=np.abs(ff["z_bin"])
count=ff["count"]
v_R_bin=ff["v_R_bin"]
v_phi_bin=np.abs(ff["v_phi_bin"])
v_z_bin=ff["v_z_bin"]
sigma_R_bin=ff["sigma_R_bin"]
sigma_phi_bin=ff["sigma_phi_bin"]
sigma_z_bin=ff["sigma_z_bin"]

try:
    Ev_R_bin=ff["Ev_R_bin"]
    Ev_phi_bin=np.abs(ff["Ev_phi_bin"])
    Ev_z_bin=ff["Ev_z_bin"]
    Esigma_R_bin=ff["Esigma_R_bin"]
    Esigma_phi_bin=ff["Esigma_phi_bin"]
    Esigma_z_bin=ff["Esigma_z_bin"]
except:
    print("no err")

    R_bin=R_bin.ravel()
    z_bin=z_bin.ravel()
    count=count.ravel()
    v_R_bin=v_R_bin.ravel()
    v_phi_bin=v_phi_bin.ravel()
    v_z_bin=v_z_bin.ravel()
    sigma_R_bin=sigma_R_bin.ravel()
    sigma_phi_bin=sigma_phi_bin.ravel()
    sigma_z_bin=sigma_z_bin.ravel()

    Ev_R_bin=np.zeros(len(v_R_bin))
    Ev_phi_bin=np.zeros(len(v_R_bin))
    Ev_z_bin=np.zeros(len(v_R_bin))
    Esigma_R_bin=np.zeros(len(v_R_bin))
    Esigma_phi_bin=np.zeros(len(v_R_bin))
    Esigma_z_bin=np.zeros(len(v_R_bin))


interp_kind = 'linear'


# 2d fit method

def f(x, a1, a2, a3, a4):
    (R, z) = x
    return 30*a1*np.exp(-a2*(R/8.122-1))*(1+(a3*z/R)**2)**a4

def uf(x, a1, a2, a3, a4):
    (R, z) = x
    return 30*a1*umath.exp(-a2*(R/8.122-1))*(1+(a3*z/R)**2)**a4

####
popt_sigma_R, pcov_sigma_R = optimize.curve_fit(f, (R_bin, z_bin), sigma_R_bin, maxfev = 10000)
print("function: 30*a1*np.exp(-a2*(R/8.122-1))*(1+(a3*z/R)**2)**a4")
print("   name             a1            a2           a3           a4")
print("sigmaR    popt:", popt_sigma_R, "\n           cov:", np.sqrt(np.diag(pcov_sigma_R)))
sigmaR = lambda R, z: f((R, z), *popt_sigma_R)*KMpStKPCpMYR
perr_sigma_R = np.sqrt(np.diag(pcov_sigma_R))
upopt_sigmaR = unumpy.uarray(popt_sigma_R, perr_sigma_R)
sigmaR_fitting_error = lambda R, z: (uf((R, z), *upopt_sigmaR)).std_dev*KMpStKPCpMYR
####
popt_sigma_phi, pcov_sigma_phi = optimize.curve_fit(f, (R_bin[count>50], z_bin[count>50]), sigma_phi_bin[count>50], maxfev = 10000)
print("sigma_phi popt:", popt_sigma_phi, "\n           cov:", np.sqrt(np.diag(pcov_sigma_phi)))
sigmaT = lambda R, z: f((R, z), *popt_sigma_phi)*KMpStKPCpMYR
perr_sigma_phi = np.sqrt(np.diag(pcov_sigma_phi))
upopt_sigmaT = unumpy.uarray(popt_sigma_phi, perr_sigma_phi)
sigmaT_fitting_error = lambda R, z: (uf((R, z), *upopt_sigmaT)).std_dev*KMpStKPCpMYR
####
popt_v_phi, pcov_v_phi = optimize.curve_fit(f, (R_bin, z_bin), v_phi_bin, maxfev = 10000, p0=[ 6.91519644, 0.00853049, 2.41782477, -0.74202121])
print("    v_phi popt:", popt_v_phi, "\n           cov:", np.sqrt(np.diag(pcov_v_phi)))
VT = lambda R, z: f((R, z), *popt_v_phi)*KMpStKPCpMYR
perr_V_phi = np.sqrt(np.diag(pcov_v_phi))
upopt_VT = unumpy.uarray(popt_v_phi, perr_V_phi)
VT_fitting_error = lambda R, z: (uf((R, z), *upopt_VT)).std_dev*KMpStKPCpMYR
####
popt_sigma_z, pcov_sigma_z = optimize.curve_fit(f, (R_bin, z_bin), sigma_z_bin, maxfev = 10000)
print("sigma_z   popt:", popt_sigma_z, "\n           cov:", np.sqrt(np.diag(pcov_sigma_z)))
sigmaz = lambda R, z: f((R, z), *popt_sigma_z)*KMpStKPCpMYR
perr_sigma_z = np.sqrt(np.diag(pcov_sigma_z))
upopt_sigmaz = unumpy.uarray(popt_sigma_z, perr_sigma_z)
sigmaz_fitting_error = lambda R, z: (uf((R, z), *upopt_sigmaz)).std_dev*KMpStKPCpMYR
####

# all error analysis
@lru_cache()
def EsigmaR(R, z):
    x = griddata(np.c_[R_bin, z_bin], Esigma_R_bin, np.c_[[R], [z]], method=interp_kind)
    if np.isnan(x):
        x = griddata(np.c_[R_bin, z_bin], Esigma_R_bin, np.c_[[R], [z]], method="nearest")
    delta_a = x*KMpStKPCpMYR
    total =  np.sqrt(delta_a**2 + sigmaR_fitting_error(R, z)**2)
    # print("ESR:", sigmaR(R, z)/KMpStKPCpMYR, x, total/KMpStKPCpMYR)
    return total

@lru_cache()
def Esigmaz(R, z):
    x = griddata(np.c_[R_bin, z_bin], Esigma_z_bin, np.c_[[R], [z]], method=interp_kind)
    if np.isnan(x):
        x = griddata(np.c_[R_bin, z_bin], Esigma_z_bin, np.c_[[R], [z]], method="nearest")
    # print("ESZ:", x)
    delta_a = x*KMpStKPCpMYR
    total = np.sqrt(delta_a**2 + sigmaz_fitting_error(R, z)**2)
    # print("ESZ:", sigmaz(R, z)/KMpStKPCpMYR, x, total/KMpStKPCpMYR)
    return total

@lru_cache()
def EsigmaT(R, z):
    x = griddata(np.c_[R_bin, z_bin], Esigma_phi_bin, np.c_[[R], [z]], method=interp_kind)
    if np.isnan(x):
        x = griddata(np.c_[R_bin, z_bin], Esigma_phi_bin, np.c_[[R], [z]], method="nearest")
    # print("EST:", x)
    delta_a = x*KMpStKPCpMYR
    total =  np.sqrt(delta_a**2 + sigmaT_fitting_error(R, z)**2)
    # print("EST:", sigmaz(R, z)/KMpStKPCpMYR, x, total/KMpStKPCpMYR)
    return total

@lru_cache()
def EVT(R, z):
    x = griddata(np.c_[R_bin, z_bin], Ev_phi_bin, np.c_[[R], [z]], method=interp_kind)
    if np.isnan(x):
        x = griddata(np.c_[R_bin, z_bin], Ev_phi_bin, np.c_[[R], [z]], method="nearest")
    # print("EVT", x)
    delta_a = x*KMpStKPCpMYR
    total = np.sqrt(delta_a**2 + VT_fitting_error(R, z)**2)
    # print("EVT:", R, z, VT(R, z)/KMpStKPCpMYR, x, total/KMpStKPCpMYR)
    return total



def alpha(R, z):
    # Everall+2019
    # alpha_0 = 0.952
    alpha_0 = 0.909
    return alpha_0*np.arctan(z/R)

    # Mackereth_Bovy_etal_2019mnras
    # return 0.9*np.arctan(np.abs(z)/R) - 0.01