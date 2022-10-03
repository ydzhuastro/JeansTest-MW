#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    分bin计算速度弥散、误差
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_dd
# 用法详见 http://scipy.github.io/devdocs/generated/scipy.stats.binned_statistic_dd.html#scipy.stats.binned_statistic_dd
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
        假设轴对称，实现对数据在柱坐标下的分bin，统计每个bin的平均速度、速度误差、速度弥散、速度弥散误差
        使用bootstrap方法计算std的ste
        
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

    # 所有点的坐标
    sample = np.c_[R_list, z_list]
    # 设置bin
    bin_R = np.arange(0, R_max, binsize_R)
    bin_z = np.arange(-z_max, z_max, binsize_z)
    bins = [bin_R, bin_z]
    # 数一数每个bin里有多少个目标
    count, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_R_list, "count", bins)
    # RR, _, _, _ = binned_statistic_2d(
    #     R_list, z_list, R_list, np.mean, bins)
    # zz, _, _, _ = binned_statistic_2d(
    #     R_list, z_list, z_list, np.mean, bins)

    # count = count.ravel()
    # RR = RR.ravel()
    # zz = zz.ravel()
    # kk = count[(count>50)*(RR>6)*(RR<11.5)*(zz>-2)*(zz<2)]
    # print(np.quantile(kk, [0.16, 0.5, 0.84]))
    # exit()

    # 平均速度 R
    v_R_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_R_list, "mean", bins)
    Ev_R_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_R_list, error_of_mean, bins)
    # 速度弥散 R
    sigma_R_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_R_list, "std", bins)
    Esigma_R_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_R_list, error_of_std, bins)

    # 平均速度 phi
    v_phi_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_phi_list, "mean", bins)
    Ev_phi_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_phi_list, error_of_mean, bins)
    # 速度弥散 phi
    sigma_phi_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_phi_list, fit_sigma_phi_b14, bins)
    Esigma_phi_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_phi_list, error_of_sigmaphi, bins)

    # 平均速度 z
    v_z_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_z_list, "mean", bins)
    Ev_z_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_z_list, error_of_mean, bins)
    # 速度弥散 z
    sigma_z_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_z_list, "std", bins)
    Esigma_z_bin, _, _, _ = binned_statistic_2d(
        R_list, z_list, v_z_list, error_of_std, bins)

    # 算出每个bin的近似中心位置
    # x, y = np.meshgrid(bin_R[:-1] + binsize_R/2, bin_z[:-1] + binsize_z/2)
    # R_bin = x
    # z_bin = y
    R_bin, _, _, _ = binned_statistic_2d(R_list, z_list, R_list, "mean", bins)
    z_bin, _, _, _ = binned_statistic_2d(R_list, z_list, z_list, "mean", bins)

    # 筛选多于10个的bin
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

    # 返回一维
    return R_bin, z_bin, count, v_R_bin, v_phi_bin, v_z_bin, sigma_R_bin, sigma_phi_bin, sigma_z_bin, Ev_R_bin, Ev_phi_bin, Ev_z_bin, Esigma_R_bin, Esigma_phi_bin, Esigma_z_bin


# main
if __name__ == "__main__":

    if not os.path.exists("./GClist.npz"):
        # 读取数据
        print("reading fits ...", end="", flush=True)
        RA_list, DEC_list, DISTANCE_list, PARALLAX_list, PMRA_list, PMDEC_list, RV_list, ERR_PARALLAX_list, ERR_PMRA_list, ERR_PMDEC_list, ERR_RV_list = read_fits_to_GC.read_fits(
            "./LMRCV1.fits")
        print("done")

        # 转换坐标
        print("converting ICRS to GC ...", end="", flush=True)
        R_list, phi_list, z_list, v_R_list, v_phi_list, v_z_list = read_fits_to_GC.ICRS_to_GC_list(
            RA_list, DEC_list, DISTANCE_list, PMRA_list, PMDEC_list, RV_list)
        # 求误差
        Ev_R_list, Ev_phi_list, Ev_z_list = velocity_uncertainty.err_v_list(RA_list, DEC_list, PARALLAX_list, PMRA_list, PMDEC_list, RV_list, ERR_PARALLAX_list, ERR_PMRA_list, ERR_PMDEC_list, ERR_RV_list)
        # Ev_R_list = np.abs(Ev_R_list)
        # Ev_phi_list = np.abs(Ev_phi_list)
        # Ev_z_list = np.abs(Ev_z_list)
        print("done")
        # 暂存转换好的坐标
        np.savez("GClist.npz", R_list=R_list, phi_list=phi_list, z_list=z_list, v_R_list=v_R_list,
                 v_phi_list=v_phi_list, v_z_list=v_z_list, Ev_R_list=Ev_R_list, Ev_phi_list=Ev_phi_list, Ev_z_list=Ev_z_list)
        print("GC list saved to ./GClist.npz")
    else:
        # 直接读取暂存的GC坐标
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

    # 计算平均速度、速度弥散
    print("binned statistics ...", end="", flush=True)
    # 3D
    # R_bin, phi_bin, z_bin, count, v_R_bin, v_phi_bin, v_z_bin, sigma_R_bin, sigma_phi_bin, sigma_z_bin = bin_statistics_3D(R_list, phi_list, z_list, v_R_list, v_phi_list, v_z_list)
    # 2D 轴对称情形
    # R_bin, z_bin, count, v_R_bin, v_phi_bin, v_z_bin, sigma_R_bin, sigma_phi_bin, sigma_z_bin = bin_statistics_2D(R_list, phi_list, z_list, v_R_list, v_phi_list, v_z_list)
    # 2D 轴对称情形 带误差
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

    # # 绘图 先检查原始的恒星位置分布
    # plt.figure(figsize=(8, 3))
    # plt.subplot(131)
    # plt.xlim(2, 25)
    # plt.ylim(-6, 6)
    # plt.hist2d(R_list, z_list, bins=30, range=[(2, 25), (-6, 6)])
    # plt.xlabel("R [kpc]")
    # plt.ylabel("z [kpc]")
    # plt.title("GC density")
    # # plt.show()

    # # 检查return格式
    # plt.subplot(132)
    # plt.xlim(2, 25)
    # plt.ylim(-6, 6)
    # # plt.scatter(R_bin, z_bin, c=(count.ravel('F')))
    # plt.pcolor(R_bin, z_bin, count)
    # plt.title("udf bin - 2D")

    # # 检查return格式
    # plt.subplot(133)
    # plt.xlim(2, 25)
    # plt.ylim(-6, 6)
    # plt.scatter(R_bin.ravel(), z_bin.ravel(), c=count.ravel(), marker='s', alpha=0.8, edgecolors="none", s=40)
    # # plt.pcolor(R_bin, z_bin, count.T)
    # plt.title("udf bin - 1D")
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(10, 3))
    # plt.subplot(131)
    # plt.title("sigma_z")
    # plt.xlabel("R [kpc]")
    # plt.ylabel("z [kpc]")
    # plt.pcolor(R_bin, z_bin, sigma_z_bin,cmap="gist_heat_r")
    # plt.colorbar(label="[km/s]")

    # plt.subplot(132)
    # plt.title("v_phi")
    # plt.xlabel("R [kpc]")
    # plt.ylabel("z [kpc]")
    # plt.pcolor(R_bin, z_bin, v_phi_bin, cmap="jet_r")
    # plt.colorbar(label="[km/s]")

    # plt.subplot(133)
    # plt.title("count")
    # plt.xlabel("R [kpc]")
    # plt.ylabel("z [kpc]")
    # plt.pcolor(R_bin, z_bin, count, norm=LogNorm(vmin=1, vmax=count.max()), cmap="jet_r")
    # plt.colorbar(label="count")
    # plt.tight_layout()
    # # plt.show()

    # R = R_bin.ravel()
    # v = v_phi_bin.ravel()
    # z = z_bin.ravel()
    # idx = np.concatenate(np.argwhere(np.logical_and(z>-1, z<1)))

    # plt.figure(figsize=(6, 3))
    # plt.scatter(R[idx], np.abs(v[idx]))
    # plt.ylim(0, 250)
    # plt.xlabel("R [kpc]")
    # plt.ylabel(r"$v_\phi$" + " [kpc]")
    # plt.show()
