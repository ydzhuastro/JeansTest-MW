#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from globalvars import KMpStKPCpMYR
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy import optimize
from functools import lru_cache

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plane_fitting import linear_2d_fit, quadratic_2d_fit

interp_kind="linear"

# Binney+2014 red clump data

R_list = np.array([
    7.61, 8.36, 7.51, 8.36, 7.48, 8.41, 7.52, 8.37
])

z_list = np.array([
    0.19, 0.19, 0.44, 0.43, 0.75, 0.75, 1.18, 1.19
])

v_phi_list = np.array([217.9, 211.4, 210.8, 207.9,
                       199.0, 200.1, 189.3, 191.2])

Ev_phi_list = np.array([4.5, 3.4, 5.3, 4.3, 6.9, 6.6, 9.8, 10.3])

sigma_phi_list = np.array([21.0199565826, 18.7207165184, 29.3542252224,
                           22.0906858838, 37.71595155, 29.1680335806, 40.9737530106, 36.8695329997])

Esigma_phi_list = np.array([0.377336447, 0.217204349, 0.633280125, 0.388500705,
                            0.382319685, 0.789547315, 0.4294982, 0.98201993])

Esigma_1_list = np.array([
    0.0, 1.1, 0.8, 0.6, 0.4, 0.2, 1.3, 2.0,
])

Esigma_3_list = np.array([
    1.0, 1.4, 1.0, 0.3, 0.5, 1.1, 0.3, 0.1,
])

A_list = np.array([
    0.872, 1.183, 0.394, 24.835, 0.212, 0.682, 0.554, 29.572, 0.211,
])

# tilt angle

def alpha(R, z): return A_list[0]*np.arctan(z/R)

def Sigma1(R, z): return 30 * \
    A_list[1]*np.exp(-A_list[2]*(R/8-1)) * \
    (1+(A_list[3]*z/R)**2)**A_list[4]*KMpStKPCpMYR

def Sigma3(R, z): return 30 * \
    A_list[5]*np.exp(-A_list[6]*(R/8-1)) * \
    (1+(A_list[7]*z/R)**2)**A_list[8]*KMpStKPCpMYR

def sigmaR(R, z):
    a = alpha(R, z)
    sigma1 = Sigma1(R, z)
    sigma3 = Sigma3(R, z)
    return np.sqrt(sigma1*sigma1*np.cos(a)*np.cos(a) + sigma3*sigma3*np.sin(a)*np.sin(a))

def sigmaz(R, z):
    a = alpha(R, z)
    sigma1 = Sigma1(R, z)
    sigma3 = Sigma3(R, z)
    return np.sqrt(sigma3*sigma3*np.cos(a)*np.cos(a) + sigma1*sigma1*np.sin(a)*np.sin(a))

sigma1 = Sigma1(R_list, z_list)
sigma3 = Sigma3(R_list, z_list)

dsigma1 = 30.0*np.exp(-A_list[2]*(R_list/8.0-1)) * np.power((1+(A_list[3]*z_list/R_list)*(A_list[3]*z_list/R_list)), A_list[4])*0.06-30.0*A_list[1]*np.exp(-A_list[2]*(R_list/8.0-1))*np.power((1+(A_list[3]*z_list/R_list)*(A_list[3]*z_list/R_list)), A_list[4])*(R_list/8.0-1)*0.004+30.0*A_list[1]*np.exp(-A_list[2]*(R_list/8.0-1))*np.power((1+(A_list[3]*z_list/R_list)*(A_list[3]*z_list/R_list)), A_list[4]-1)*A_list[4]*z_list*z_list/R_list/R_list*2.0*A_list[3]*0.448+30.0*A_list[1]*np.exp(-A_list[2]*(R_list/8.0-1))*np.power((1+(A_list[3]*z_list/R_list)*(A_list[3]*z_list/R_list)), A_list[4])*np.log(1+(A_list[3]*z_list/R_list)*(A_list[3]*z_list/R_list))*0.002
dsigma3 = 30.0*np.exp(-A_list[6]*(R_list/8.0-1))*np.power((1+(A_list[7]*z_list/R_list)*(A_list[7]*z_list/R_list)), A_list[8])*0.031-30.0*A_list[5]*np.exp(-A_list[6]*(R_list/8.0-1))*np.power((1+(A_list[7] * z_list/R_list)*(A_list[7]*z_list/R_list)), A_list[8])*(R_list/8.0-1)*0.192+30.0*A_list[5]*np.exp(-A_list[6]*(R_list/8.0-1))*np.power((1+(A_list[7]*z_list/R_list)*(A_list[7]*z_list/R_list)), A_list[8]-1)*A_list[8]*z_list*z_list/R_list/R_list*2.0*A_list[3]*5.243+30.0*A_list[5]*np.exp(-A_list[6]*(R_list/8.0-1))*np.power((1+(A_list[7]*z_list/R_list)*(A_list[7]*z_list/R_list)), A_list[8])*np.log(1+(A_list[7]*z_list/R_list)*(A_list[7]*z_list/R_list))*0.007
ErrSigmaR = KMpStKPCpMYR*(np.sqrt(Esigma_1_list*Esigma_1_list + dsigma1*dsigma1)*sigma1*np.cos(alpha(R_list, z_list))*np.cos(alpha(R_list, z_list))
                          + np.sqrt(Esigma_3_list*Esigma_3_list + dsigma3*dsigma3)*sigma3*np.sin(alpha(R_list, z_list))*np.sin(alpha(R_list, z_list)))/sigmaR(R_list, z_list)*np.sqrt(1.0)
ErrSigmaz = KMpStKPCpMYR*(np.sqrt(Esigma_3_list*Esigma_3_list + dsigma3*dsigma3)*sigma3*np.cos(alpha(R_list, z_list))*np.cos(alpha(R_list, z_list))
                          + np.sqrt(Esigma_1_list*Esigma_1_list + dsigma1*dsigma1)*sigma1*np.sin(alpha(R_list, z_list))*np.sin(alpha(R_list, z_list)))/sigmaz(R_list, z_list)*np.sqrt(1.0)

@lru_cache()
def EsigmaR(R, z):
    # print("ESR:", griddata(np.c_[R_list, z_list], ErrSigmaR, np.c_[[R], [z]], method=interp_kind)/KMpStKPCpMYR)
    return griddata(np.c_[R_list, z_list], ErrSigmaR, np.c_[[R], [z]], method=interp_kind)

@lru_cache()
def Esigmaz(R, z):
    # print("ESz:", griddata(np.c_[R_list, z_list], ErrSigmaz, np.c_[[R], [z]], method=interp_kind)/KMpStKPCpMYR)
    return griddata(np.c_[R_list, z_list], ErrSigmaz, np.c_[[R], [z]], method=interp_kind)

@lru_cache()
def sigmaT(R, z):
    # print("ST:", griddata(np.c_[R_list, z_list], sigma_phi_list, np.c_[[R], [z]], method=interp_kind))
    return griddata(np.c_[R_list, z_list], sigma_phi_list, np.c_[[R], [z]], method=interp_kind)*KMpStKPCpMYR

@lru_cache()
def EsigmaT(R, z):
    # print("EST:", griddata(np.c_[R_list, z_list], Esigma_phi_list, np.c_[[R], [z]], method=interp_kind))
    return griddata(np.c_[R_list, z_list], Esigma_phi_list, np.c_[[R], [z]], method=interp_kind)*KMpStKPCpMYR

@lru_cache()
def VT(R, z):
    # print("VT:", griddata(np.c_[R_list, z_list], v_phi_list, np.c_[[R], [z]], method=interp_kind))
    return griddata(np.c_[R_list, z_list], v_phi_list, np.c_[[R], [z]], method=interp_kind)*KMpStKPCpMYR

@lru_cache()
def EVT(R, z):
    # print("EVT:", griddata(np.c_[R_list, z_list], Ev_phi_list, np.c_[[R], [z]], method=interp_kind))
    return griddata(np.c_[R_list, z_list], Ev_phi_list, np.c_[[R], [z]], method=interp_kind)*KMpStKPCpMYR