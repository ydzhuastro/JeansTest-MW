#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Calculate mean and dispersion of velocities
    Author: Yongda Zhu (yzhu144@ucr.edu), Hai-Xia Ma
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_dd
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit

from uncertainties import ufloat
from uncertainties import umath 
from uncertainties import unumpy

from astropy.stats import bootstrap
from scipy import stats
from matplotlib.colors import LogNorm
import read_fits_to_GC
import velocity_uncertainty

def fit_sigma_phi_b14(v_phi):
    """
        fit the profile of Eqs 7&8 in Binney+2014
        Args:
            x: array of vphi
        Return:
            sigma_phi: velocity dispersion at the mean velocity
            esigma_phi: error of sigma_phi
    """
    v_phi = np.abs(v_phi)
    def sigma(v, b1, b2, b3, b4):
        v100 = v/100
        return b1 + b2*v100 + b3*v100**2 + b4*v100**3

    def profile(v, b0, b1, b2, b3, b4):
        v100 = v/100
        s = b1 + b2*v100 + b3*v100**2 + b4*v100**3
        return np.exp(- (v - b0)**2/2/s**2)

    if len(v_phi) < 50:
        return np.std(v_phi)
    else:
        try:
            p, v = np.histogram(v_phi, bins=17)
            p = p/np.max(p)
            v = (v[1:]+v[:-1])/2
            popt, _ = curve_fit(profile, v, p, p0=[200, 30, 0, 0, 0], maxfev=10000)
            mv = np.mean(v_phi)
            # perr = np.sqrt(np.diag(pcov))
            b0, b1, b2, b3, b4 = popt
            sigma_phi = sigma(mv, b1, b2, b3, b4)
            std = np.std(v_phi)
            # print(np.std(v_phi), sigma_phi)
            if False:
                plt.plot(v, p, ds="steps-mid")
                plt.plot(v, profile(v, *popt), ds="steps-mid")
                plt.show(block=False)
                plt.pause(0.01)
                plt.close()
            if np.abs(std - sigma_phi) > 0.5*std:
                return std
            else:
                return sigma_phi
        except:
            return np.std(v_phi)


def bin_statistics_2D_ERR(R_list, phi_list, z_list, v_R_list, v_phi_list, v_z_list, Ev_R_list, Ev_phi_list, Ev_z_list, binsize_R=0.2, binsize_z=0.05, R_max=30, z_max=6):
    """
        bin the data in cylindrical coordinates and calculate the mean and dispersions of velocities
        
        Args:
            R_list, phi_list, z_list, v_R_list, v_phi_list, v_z_list
        Returns:
            R_bin, z_bin, count, v_R_bin, v_phi_bin, v_z_bin, sigma_R_bin, sigma_phi_bin, sigma_z_bin
    """

    def error_of_mean(x):
        # if len(error_array) > 0:
        #     return np.sqrt(np.sum(error_array**2)) / error_array.size
        # else:
        #     return np.nan
        if len(x) > 1:
            boot = bootstrap(x, 100, bootfunc=np.mean)
            # return np.std(boot)/np.sqrt(len(x)-1)
            return stats.sem(boot)
        else:
            return np.nan

    def error_of_std(x):
        if len(x) > 1:
            boot = bootstrap(x, 100, bootfunc=np.std)
            # return np.std(boot)/np.sqrt(len(x)-1)
            return stats.sem(boot)
        else:
            return np.nan

    def error_of_sigmaphi(x):
        if len(x) > 1:
            boot = bootstrap(x, 100, bootfunc=fit_sigma_phi_b14)
            # return np.std(boot)/np.sqrt(len(x)-1)
            return stats.sem(boot)
        else:
            return np.nan

    count_threshold = 10

    sample = np.c_[R_list, z_list]
    # set bins
    bin_R = np.arange(0, R_max, binsize_R)
    bin_z = np.arange(-z_max, z_max, binsize_z)
    bins = [bin_R, bin_z]
    # count objects in each bin
    count, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_R_list, "count", bins)

    # <v_R>
    v_R_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_R_list, "mean", bins)
    Ev_R_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_R_list, error_of_mean, bins)
    # sigma_R
    sigma_R_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_R_list, "std", bins)
    Esigma_R_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_R_list, error_of_std, bins)

    # <v_phi>
    v_phi_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_phi_list, "mean", bins)
    Ev_phi_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_phi_list, error_of_mean, bins)
    # sigma_phi
    sigma_phi_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_phi_list, fit_sigma_phi_b14, bins)
    Esigma_phi_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_phi_list, error_of_sigmaphi, bins)

    # <v_z>
    v_z_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_z_list, "mean", bins)
    Ev_z_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_z_list, error_of_mean, bins)
    # <sigma_z>
    sigma_z_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_z_list, "std", bins)
    Esigma_z_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_z_list, error_of_std, bins)

    # center of each bin
    R_bin, _, _, _ = binned_statistic_2d(R_list, z_list, R_list, "mean", bins)
    z_bin, _, _, _ = binned_statistic_2d(R_list, z_list, z_list, "mean", bins)

    # filter bins with too few objects
    count = count.ravel()
    idx = np.concatenate(np.argwhere(count > count_threshold))
    count = count[idx]

    R_bin = R_bin.ravel()[idx]
    z_bin = z_bin.ravel()[idx]
    v_R_bin = v_R_bin.ravel()[idx]
    v_phi_bin = v_phi_bin.ravel()[idx]
    v_z_bin = v_z_bin.ravel()[idx]
    sigma_R_bin = sigma_R_bin.ravel()[idx]
    sigma_phi_bin = sigma_phi_bin.ravel()[idx]
    sigma_z_bin = sigma_z_bin.ravel()[idx]

    Ev_R_bin = Ev_R_bin.ravel()[idx]
    Ev_phi_bin = Ev_phi_bin.ravel()[idx]
    Ev_z_bin = Ev_z_bin.ravel()[idx]
    Esigma_R_bin = Esigma_R_bin.ravel()[idx]
    Esigma_phi_bin = Esigma_phi_bin.ravel()[idx]
    Esigma_z_bin = Esigma_z_bin.ravel()[idx]

    # return 1D array
    return R_bin, z_bin, count, v_R_bin, v_phi_bin, v_z_bin, sigma_R_bin, sigma_phi_bin, sigma_z_bin, Ev_R_bin, Ev_phi_bin, Ev_z_bin, Esigma_R_bin, Esigma_phi_bin, Esigma_z_bin


# main
if __name__ == "__main__":

    if not os.path.exists("./GClist.npz"):
        # read fits
        print("reading fits ...", end="", flush=True)
        RA_list, DEC_list, DISTANCE_list, PARALLAX_list, PMRA_list, PMDEC_list, RV_list, ERR_PARALLAX_list, ERR_PMRA_list, ERR_PMDEC_list, ERR_RV_list = read_fits_to_GC.read_fits(
            "./LMRCV1.fits")
        print("done")

        # convert corrdinates
        print("converting ICRS to GC ...", end="", flush=True)
        R_list, phi_list, z_list, v_R_list, v_phi_list, v_z_list = read_fits_to_GC.ICRS_to_GC_list(
            RA_list, DEC_list, DISTANCE_list, PMRA_list, PMDEC_list, RV_list)
        # evaluate uncertainties
        Ev_R_list, Ev_phi_list, Ev_z_list = velocity_uncertainty.err_v_list(RA_list, DEC_list, PARALLAX_list, PMRA_list, PMDEC_list, RV_list, ERR_PARALLAX_list, ERR_PMRA_list, ERR_PMDEC_list, ERR_RV_list)
        # Ev_R_list = np.abs(Ev_R_list)
        # Ev_phi_list = np.abs(Ev_phi_list)
        # Ev_z_list = np.abs(Ev_z_list)
        print("done")
        # save GC corrdinates list
        np.savez("GClist.npz", R_list=R_list, phi_list=phi_list, z_list=z_list, v_R_list=v_R_list,
                 v_phi_list=v_phi_list, v_z_list=v_z_list, Ev_R_list=Ev_R_list, Ev_phi_list=Ev_phi_list, Ev_z_list=Ev_z_list)
        print("GC list saved to ./GClist.npz")
    else:
        # read cached GC list
        print("reading GClist from file ... ", end="", flush=True)
        f = np.load("./GClist.npz")
        R_list = f["R_list"]
        phi_list = f["phi_list"]
        z_list = f["z_list"]
        v_R_list = f["v_R_list"]
        v_phi_list = f["v_phi_list"]
        v_z_list = f["v_z_list"]
        Ev_R_list = f["Ev_R_list"]
        Ev_phi_list = f["Ev_phi_list"]
        Ev_z_list = f["Ev_z_list"]
        print("done")
        print("%d objects loaded!" % len(R_list))

    # calculate the <v> and sigma
    print("binned statistics ...", end="", flush=True)
    R_bin, z_bin, count, v_R_bin, v_phi_bin, v_z_bin, sigma_R_bin, sigma_phi_bin, sigma_z_bin, Ev_R_bin, Ev_phi_bin, Ev_z_bin, Esigma_R_bin, Esigma_phi_bin, Esigma_z_bin = bin_statistics_2D_ERR(
        R_list, phi_list, z_list, v_R_list, v_phi_list, v_z_list, Ev_R_list, Ev_phi_list, Ev_z_list)
    print("done")
    print("(z, R bins): ", R_bin.shape)

    # write to file
    filename = "./binned_results.npz"
    # np.savez(filename, R_bin=R_bin, z_bin=z_bin, count=count, v_R_bin=v_R_bin, v_phi_bin=v_phi_bin, v_z_bin=v_z_bin, sigma_R_bin=sigma_R_bin, sigma_phi_bin=sigma_phi_bin, sigma_z_bin=sigma_z_bin)
    np.savez(filename, R_bin=R_bin, z_bin=z_bin, count=count, 
            v_R_bin=v_R_bin, v_phi_bin=v_phi_bin,
            v_z_bin=v_z_bin, sigma_R_bin=sigma_R_bin, sigma_phi_bin=sigma_phi_bin, sigma_z_bin=sigma_z_bin,
            Ev_R_bin=Ev_R_bin, Ev_phi_bin=Ev_phi_bin,
            Ev_z_bin=Ev_z_bin, Esigma_R_bin=Esigma_R_bin, Esigma_phi_bin=Esigma_phi_bin, Esigma_z_bin=Esigma_z_bin)
    print("results saved to " + filename)
