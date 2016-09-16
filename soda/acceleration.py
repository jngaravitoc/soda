"""
This code computes the acceleration of the host galaxy
and the satellite galaxy.

TO-DO:
------
    1. Generalize to N-satellites.
"""

import numpy as np
from astropy import units, constants


#Function that computes the satellite acceleration
def acc_sat(xyz, vxyz, host_model, sat_model, disk_params, \
            bulge_params, ac, dfric, alpha=False):
    """
    Function that computes the satellite acceleration
    due to it's host galaxy.

    Input:
    ------
    xyz:
    vxyz:
    host_model:
    sat_model:
    disk_params:
    bulge_params:
    ac default(0): Adiabatic contraction. No(0), Yes(1)
    dfric default(1): dynamical friction. No(0), Yes(1).
    alpha: cl (Coulomb logarithm (0:), (1:) ), alpha, L if cl=, C if cl = 

    Output:
    -------

    Dependencies:
    -------------
    Astropy

    TO-DO:
    ------
    1. Disk and Bulge as optional parameters.
    2. Link with galpy/gala
    """

    r = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)

    # Host & Satellite models & parameters
    M_host = host_model[1]
    M_sat = sat_model[1]
    Rvir_host = host_model[2]
    # Disk and bulge parameters
    M_disk, a_disk, b_disk = disk_params
    M_bulge, rh = bulge_params
    # Acceleration by the DM halo profile

    if ((r<=Rvir_host) & (dfric==1)):
        if (ac == 1):
            print('No ac yet!')
            #ahalo = acc_ac(x, y, z)
        else:
            if (host_model[0] == 'NFW'):
                Rvir_host = host_model[2]
                c_host = host_model[3]
                ahalo = a_NFWnRvir(c_host, xyz[0], xyz[1], xyz[2],\
                                   M_host, Rvir_host)

            elif (host_model[0] == 'hernquist'):
                rs_host = host_model[2]
                ahalo = a_hernquist(rs_host, xyz[0], xyz[1], xyz[2],\
                                    M_host)

        adisk = a_mn(a_disk, b_disk, xyz[0], xyz[1], xyz[2], M_disk)
        abulge = a_hernquist(rh, xyz[0], xyz[1], xyz[2], M_bulge)

        ax = ahalo[0] + adisk[0] + abulge[0]
        ay = ahalo[1] + adisk[1] + abulge[1]
        az = ahalo[2] + adisk[2] + abulge[2]

        # Truncating the halo at r_vir:
        # Dynamical Friction inside the r_vir
        a_dfx, a_dfy, a_dfz = df(xyz[0], xyz[1], xyz[2], vxyz[0], vxyz[1], \
                                 xyz[2], M_host, M_sat, Rvir_host, c_host, \
                                 host_model, M_disk, M_bulge, ac, alpha)
        Ax = ax + a_dfx
        Ay = ay + a_dfy
        Az = az + a_dfz

    #Point like acceleration beyond r_vir
    else:
        Mtot = (M_host + M_disk + M_bulge) * units.Msun
        Ax = - G * Mtot * xyz[0] * units.kpc / (r*units.kpc)**3
        Ay = - G * Mtot * xyz[1] * units.kpc / (r*units.kpc)**3
        Az = - G * Mtot * xyz[2] * units.kpc / (r*units.kpc)**3
        Ax = Ax.to(units.kpc / units.Gyr**2).value
        Ay = Ay.to(units.kpc / units.Gyr**2).value
        Az = Az.to(units.kpc / units.Gyr**2).value
        #x = Ax.value
        #Ay = Ay.value
        #Az = Az.value
    return Ax, Ay, Az

def acc_host(xyz, vxyz, host_model, sat_model):

    """
    Function that computes the host galaxy  acceleration
    due to its satellite.

    Input:
    ------
    xyz:
    vxyz:
    host_model:
    sat_model:


    Output:
    -------

    Dependencies:
    -------------
    Astropy

    TO-DO:
    ------
    1. Link with galpy/gala
    2. Dynamical Friction of the host.
    """

    M_sat = sat_model[1]

    if (sat_model[0] == 'NFW'):
        A_host = a_NFWnRvir(c_sat, xyz[0], xyz[1], xyz[2], M_sat, Rvir_sat)
        M_sat = sat_model[1]
        Rvir_sat = sat_model[2]
        c_sat = sat_model[3]

    elif (sat_model[0] == 'hernquist'):
        M_sat = sat_model[1]
        rs_sat = sat_model[2]
        A_host = a_hernquist(rs_sat, xyz[0], xyz[1], xyz[2], M_sat)

    elif (sat_model[0] == 'plummer'):
        M_sat = sat_model[1]
        rs_sat = sat_model[2]
        A_host = a_plummer(rs_sat, xyz[0], xyz[1], xyz[2], M_sat)

    Ax = A_host[0]
    Ay = A_host[1]
    Az = A_host[2]

    """
    # Host dynamical friction:
    if (Host_df==1):
        D = np.sqrt(x**2 + y**2 + z**2)
        R_mass = Rvir_host - (Rvir_sat - D)
        # Mass fraction of the host galaxy inside the satellite.
        if (Host_model == 0):
            M_frac = mass_NFWnRvir(c_host, R_mass, 0, 0, M_host, Rvir_host)
        elif (Host_model ==1):
            M_frac = mass_hernquist(rs_host, R_mass,M_host)
        a_dfx, a_dfy, a_dfz = df(x, y, z, vx, vy, vz, M_sat, M_frac,\
                              Rvir_sat, c_sat, alpha_df_host)
        Ax = Ax + a_dfx
        Ay = Ay + a_dfy
        Az = Az + a_dfz
    """
    return Ax, Ay, Az
