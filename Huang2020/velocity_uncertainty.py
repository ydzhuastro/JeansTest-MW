#!/usr/bin/env python
# -*- coding: utf-8

"""
    Propagating uncertainties from proper motion to velocities in GC coordinates.
    Reference: https://faculty.virginia.edu/ASTR5610/lectures/VELOCITIES/velocities.html
"""

import numpy as np
import numpy.matlib
from functools import lru_cache
from joblib import delayed, Parallel

alpha_NGP = 192.25/180*np.pi
delta_NGP = 27.4/180*np.pi
theta_0 = 123/180*np.pi

T = np.matlib.matrix([[-0.06699, -0.87276, -0.48354], [0.49273, -0.45035, 0.74458], [-0.86760, -0.18837, 0.46020]])

@lru_cache()
def A(alpha, delta):
    return np.matlib.matrix([
        [np.cos(alpha)*np.cos(delta), -np.sin(alpha), -np.cos(alpha)*np.sin(delta)],
        [np.sin(alpha)*np.cos(delta), np.cos(alpha), -np.sin(alpha)*np.sin(delta)],
        [np.sin(delta), 0, np.cos(delta)]
    ])

@lru_cache()
def B(alpha, delta):
    return T*A(alpha, delta)

@lru_cache()
def C(alpha, delta):
    C = np.power(B(alpha, delta), 2)
    return C

def err_v(RA, DEC, PARALLAX, PMRA, PMDEC, RV, EPARALLAX, EPMRA, EPMDEC, ERV):
    """
        RA, DEC: deg
        PARALLAX, E: mas
        PMRA, PMDEC, E: mas/yr
        RV, E: km/s
    """
    pi = PARALLAX/1000
    sigma_pi = EPARALLAX/1000
    rho = RV
    sigma_rho = ERV
    u_alpha = PMRA/1000
    sigma_u_alpha = EPMRA/1000
    u_delta = PMDEC/1000
    sigma_u_delta = EPMDEC/1000

    alpha = RA/180*np.pi
    delta = DEC/180*np.pi

    b = B(alpha, delta)
    c = C(alpha, delta)
    k = 4.74057

    mat_sigma_v2 = c*np.matlib.matrix([
        [sigma_rho**2],
        [(k/pi)**2*( sigma_u_alpha**2 + (u_alpha*sigma_pi/pi)**2 )],
        [(k/pi)**2*( sigma_u_delta**2 + (u_delta*sigma_pi/pi)**2 )]
    ]) + 2*u_alpha*u_delta*k**2*sigma_pi**2/pi**4*np.matlib.matrix([
        [b[0, 1]*b[0, 2]],
        [b[1, 1]*b[1, 2]],
        [b[2, 1]*b[2, 2]]
    ])

    sigma_U = np.sqrt(mat_sigma_v2[0, 0])
    sigma_V = np.sqrt(mat_sigma_v2[1, 0])
    sigma_W = np.sqrt(mat_sigma_v2[2, 0])
    return sigma_U, sigma_V, sigma_W

def err_v_list(RA_list, DEC_list, PARALLAX_list, PMRA_list, PMDEC_list, RV_list, EPARALLAX_list, EPMRA_list, EPMDEC_list, ERV_list):
    print("  [ICRS_to_GC_V_err] " + "total: ", len(RA_list))

    # for i in range(len(RA_list)):
    #     Ev_R, Ev_phi, Ev_z = err_v(RA_list[i], DEC_list[i], PARALLAX_list[i], PMRA_list[i], PMDEC_list[i], RV_list[i], EPARALLAX_list[i], EPMRA_list[i], EPMDEC_list[i], ERV_list[i])
    #     print(Ev_R, Ev_phi, Ev_z)
    # exit()

    # tuple_list = list(zip(*[ICRS_to_GC(RA, DEC, DISTANCE, PMRA, PMDEC, RV) for RA, DEC, DISTANCE, PMRA, PMDEC, RV in zip(RA_list, DEC_list, DISTANCE_list, PMRA_list, PMDEC_list, RV_list)]))
    # parallel version 8 cores, about 6 mins to run:
    values = Parallel(n_jobs=8)(delayed(err_v)(RA, DEC, PARALLAX, PMRA, PMDEC, RV, EPARALLAX, EPMRA, EPMDEC, ERV) for RA, DEC, PARALLAX, PMRA, PMDEC, RV, EPARALLAX, EPMRA, EPMDEC, ERV in zip(RA_list, DEC_list, PARALLAX_list, PMRA_list, PMDEC_list, RV_list, EPARALLAX_list, EPMRA_list, EPMDEC_list, ERV_list))
    tuple_list = list(zip(*values))

    print("  [ICRS_to_GC_list] " + "converting")
    # using list comprehension to convert tuple into list
    Ev_R_list, Ev_phi_list, Ev_z_list = [list(t) for t in tuple_list]

    return Ev_R_list, Ev_phi_list, Ev_z_list
