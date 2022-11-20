#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Converting ICRS to GC corrdinates;
    Please check the values of UVW at the line of `v_sun`
    Author: Yongda Zhu (yzhu144@ucr.edu), Hai-Xia Ma
    ver. Jun 23, 2020
"""

import astropy.units as u
import astropy.coordinates as apycoords
import numpy as np
from astropy.io import fits

from joblib import delayed, Parallel

filename='LMRCV1.fits'

def read_fits(filename):
    hdul=fits.open(filename)

    RA_list=hdul[1].data.field('RA')
    DEC_list=hdul[1].data.field('DEC')
    DISTANCE_list=hdul[1].data.field('DISTANCE')
    PARALLAX_list = hdul[1].data.field('PARALLAX')
    PMRA_list=hdul[1].data.field('PMRA')
    PMDEC_list=hdul[1].data.field('PMDEC')
    RV_list=hdul[1].data.field('RV')
    ERR_PARALLAX_list = hdul[1].data.field('ERR_PARALLAX')
    ERR_PMRA_list=hdul[1].data.field('ERR_PMRA')
    ERR_PMDEC_list=hdul[1].data.field('ERR_PMDEC')
    ERR_RV_list=hdul[1].data.field('ERR_RV')
    FEH_list=hdul[1].data.field('FEH')
    AFE_list=hdul[1].data.field('AFE')         

    return RA_list, DEC_list, DISTANCE_list, PARALLAX_list, PMRA_list, PMDEC_list, RV_list, ERR_PARALLAX_list, ERR_PMRA_list, ERR_PMDEC_list, ERR_RV_list

def ICRS_to_GC(RA, DEC, DISTANCE, PMRA, PMDEC, RV):
    """
        Arg: 
            RA, DEC, DISTANCE, PMRA, PMDEC, RV
        Return:
            R, phi, z, v_R, v_phi, v_z
    """

    # unpack params
    # RA, DEC, DISTANCE, PMRA, PMDEC, RV = input_list
    # ICRS coordinates
    c = apycoords.SkyCoord(ra=RA*u.deg, dec=DEC*u.deg, distance=DISTANCE*u.kpc, frame='icrs',
                           radial_velocity=RV*u.km/u.s, pm_ra_cosdec=PMRA*u.mas/u.yr, pm_dec=PMDEC*u.mas/u.yr)
    # Define GC frame
    # Schönrich et al. (2010), Binney et al. (2014)
    v_sun = apycoords.CartesianDifferential([11.1, 232.24, 7.25]*u.km/u.s)
    gc_frame = apycoords.Galactocentric(galcen_distance=8.0*u.kpc,
                                        # z_sun=25.*u.pc,
                                        galcen_v_sun=v_sun)
    cg=c.transform_to(gc_frame)
    cg.representation_type = 'cylindrical'
    R = cg.rho.to(u.kpc).value
    phi = cg.phi.to(u.deg).value
    z = cg.z.to(u.kpc).value
    v_R = cg.d_rho.to(u.km/u.s).value
    v_phi = (cg.d_phi*cg.rho).to(u.km/u.s, equivalencies=u.dimensionless_angles()).value
    v_z = cg.d_z.to(u.km/u.s).value

    # output_list = [R, phi, z, v_R, v_phi, v_z]

    return R, phi, z, v_R, v_phi, v_z

def _test_ICRS_to_GC():
    ra= 120.#u.deg
    dec= 30.#u.deg
    distance= 1.2#u.kpc
    pmra= 5.#u.mas/u.yr
    pmdec= -3.#u.mas/u.yr
    vlos= 55.#u.km/u.s
    
    R, phi, z, v_R, v_phi, v_z = ICRS_to_GC(ra, dec, distance, pmra, pmdec, vlos)
    print(R, phi, z, v_R, v_phi, v_z)

def ICRS_to_GC_list(RA_list, DEC_list, DISTANCE_list, PMRA_list, PMDEC_list, RV_list):
    """
        ICRS_to_GC for a list of objects
        Args: 
            RA, DEC, DISTANCE, PMRA, PMDEC, RV : list
        Returns:
            R, phi, z, v_R, v_phi, v_z : list
    """
    # notes on zip(*list):
    # >>> l = [(1,2), (3,4), (8,9)]
    # >>> list(zip(*l))
    # [(1, 3, 8), (2, 4, 9)]
    print("  [ICRS_to_GC_list] " + "total: ", len(RA_list))

    # tuple_list = list(zip(*[ICRS_to_GC(RA, DEC, DISTANCE, PMRA, PMDEC, RV) for RA, DEC, DISTANCE, PMRA, PMDEC, RV in zip(RA_list, DEC_list, DISTANCE_list, PMRA_list, PMDEC_list, RV_list)]))
    # parallel version 8 cores, about 6 mins to run:
    values = Parallel(n_jobs=8)(delayed(ICRS_to_GC)(RA, DEC, DISTANCE, PMRA, PMDEC, RV) for RA, DEC, DISTANCE, PMRA, PMDEC, RV in zip(RA_list, DEC_list, DISTANCE_list, PMRA_list, PMDEC_list, RV_list))
    tuple_list = list(zip(*values))

    print("  [ICRS_to_GC_list] " + "converting")
    # then we use list comprehension to convert tuple to list
    R_list, phi_list, z_list, v_R_list, v_phi_list, v_z_list = [list(t) for t in tuple_list]

    return R_list, phi_list, z_list, v_R_list, v_phi_list, v_z_list

def _test_ICRS_to_GC_list():
    ra_list= [120.]*3#u.deg
    dec_list= [30.]*3#u.deg
    distance_list= [1.2]*3#u.kpc
    pmra_list= [5.]*3#u.mas/u.yr
    pmdec_list= [-3.]*3#u.mas/u.yr
    vlos_list= [55.]*3#u.km/u.s
    
    R, phi, z, v_R, v_phi, v_z = ICRS_to_GC_list(ra_list, dec_list, distance_list, pmra_list, pmdec_list, vlos_list)
    print(R, phi, z, v_R, v_phi, v_z)

if __name__ == "__main__":
    _test_ICRS_to_GC_list()

