"""
This code computes the acceleration of the host galaxy
and the satellite galaxy.

TO-DO:
------
    1. Generalize to N-satellites.
"""

import numpy as np
from astropy import units, constants
from .profiles import *
from .dynamical_friction import *


def extract(dct, namespace=None):
    # function that extracts variables from kwargs
    # from:
    # http://stackoverflow.com/questions/4357851/creating-or-assigning-variables-from-a-dictionary-in-python

    if not namespace: namespace = globals()
    namespace.update(dct)

def particle_acceleration_LMC(xyz_LMC, xyz_MW, sat_model, host_model, \
                              disk_params, bulge_params, ac, **kwargs):


    """
    Acceleration of a particle in the presence of the LMC,
    this neglects the extended nature of the LMC, i.e the
    particle always feels the LMC potential as that of a
    point-like potential.

    """

    extract(kwargs)

    M_LMC = sat_model[1] * units.Msun
    r_to_LMC = np.sqrt(xyz_LMC[0]**2.0 + xyz_LMC[1]**2.0 + xyz_LMC[2]**2.0)
    Ax_LMC, Ay_LMC, Az_LMC = particle_acceleration(M_LMC, xyz_LMC, r_to_LMC)
    Ax_MW, Ay_MW, Az_MW = acc_sat_helper(xyz_MW, host_model, ac,
                                         disk_params=disk_params, bulge_params=bulge_params)

    if 'pos_sat2' in kwargs:
        M_sag = sat_model2[1]*units.Msun
        r_to_sag = np.sqrt(pos_sat2[0]**2.0 + pos_sat2[1]**2.0 + pos_sat2[2]**2.0)
        Ax_sag, Ay_sag, Az_sag = particle_acceleration(M_sag, pos_sat2, r_to_sag)
        return Ax_LMC + Ax_MW + Ax_sag, Ay_LMC + Ay_MW + Ay_sag, Az_LMC + Az_MW + Az_sag
    else:
        return Ax_LMC + Ax_MW, Ay_LMC + Ay_MW, Az_LMC + Az_MW



def particle_acceleration(Mtot, xyz, r):
    """
    Newtonian acceleration between two bodies.

    """
    Ax = - G * Mtot * xyz[0] * units.kpc / (r*units.kpc)**3
    Ay = - G * Mtot * xyz[1] * units.kpc / (r*units.kpc)**3
    Az = - G * Mtot * xyz[2] * units.kpc / (r*units.kpc)**3
    Ax = Ax.to(units.kpc / units.Gyr**2).value
    Ay = Ay.to(units.kpc / units.Gyr**2).value
    Az = Az.to(units.kpc / units.Gyr**2).value

    return Ax, Ay, Az


def acc_sat_helper(xyz, host_model, ac,
                   **kwargs):
    """
    Computes the acceleration of a particle inside a given MW model.
    """

    extract(kwargs)

    if 'disk_params' in kwargs:
        M_disk, a_disk, b_disk = disk_params

    if 'bulge_params' in kwargs:
        M_bulge, rh = bulge_params

    M_host = host_model[1]



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


        elif (host_model[0] == 'NFW_T'):

            #v_halo = host_model[1]
            q = host_model[6]
            s = host_model[7]
            ahalo = a_NFWnRvir(c_host, xyz[0], xyz[1], xyz[2],\
                               M_host, Rvir_host, q, s)

    Ax = ahalo[0]
    Ay = ahalo[1]
    Az = ahalo[2]


    if 'disk_params' in kwargs:
        adisk = a_mn(a_disk, b_disk, xyz[0], xyz[1], xyz[2], M_disk)
        Ax += adisk[0]
        Ay += adisk[1]
        Az += adisk[2]

    if 'bulge_params' in kwargs:
        abulge = a_hernquist(rh, xyz[0], xyz[1], xyz[2], M_bulge)
        Ax += abulge[0]
        Ay += abulge[1]
        Az += abulge[2]


    if 'sat2' in kwargs:
        ax_s2, ay_s2, az_s2 = particle_acceleration(Msat2, xyzrs1rs2, rs1s2)
        Ax+=ax_s2
        Ay+=ay_s2
        Az+=az_s2

    return Ax, Ay, Az

#Function that computes the satellite acceleration
def acc_sat(xyz, vxyz, host_model, sat_model, disk_params, \
            bulge_params, ac, dfric, alpha=False, **kwargs):
    """
    Function that computes the satellite acceleration
    due to it's host galaxy.

    Input:
    ------
    xyz:
    vxyz:
    host_model: ['halo model', Mhost, Rvir/a, c]
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
    1. Disk and Bulge as optional parameters
    2. Link with galpy/gala

    """

    extract(kwargs)

    r = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)

    # Host & Satellite models & parameters
    M_host = host_model[1]
    M_sat = sat_model[1]

    # This is not the case for the herquist profile
    Rvir_host = host_model[2] # *************************

    # Disk and bulge parameters
    M_disk, a_disk, b_disk = disk_params
    M_bulge, rh = bulge_params

    # Acceleration by the DM halo profile

    if ((r<=Rvir_host) & (dfric==1)):

        Ax, Ay, Az = acc_sat_helper(xyz, host_model, ac, disk_params=disk_params,\
                                    bulge_params=bulge_params)




        #  generalize this to a Hernquist halo as well.
        if dfric==1:
            c_host = host_model[3]
            a_dfx, a_dfy, a_dfz = df(xyz[0], xyz[1], xyz[2],\
                                     vxyz[0], vxyz[1], xyz[2],\
                                     M_host, M_sat, Rvir_host,\
                                     c_host, host_model, M_disk,\
                                     M_bulge, ac, alpha)
            Ax+=a_dfx
            Ay+=a_dfy
            Az+=a_dfz

    #Point like acceleration beyond r_vir

    else:
        Ax, Ay, Az = particle_acceleration((M_host + M_disk + \
                                           M_bulge)*units.Msun,\
                                           xyz, r)

    if 'sat2_model' in kwargs:
        r_sat2 = (xyz2[0]**2.0 + xyz2[1]**2 + xyz2[2]**2)**0.5
        Ax2, Ay2, Az2 = particle_acceleration(sat2_model[1]*units.Msun,\
                         xyz2, r_sat2)
        Ax+=Ax2
        Ay+=Ay2
        Az+=Az2

    return Ax, Ay, Az


def acc_host(xyz, vxyz, host_model, sat_model, **kwargs):

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
        c_sat = sat_model[3]
        M_sat = sat_model[1]
        Rvir_sat = sat_model[2]
        A_host = a_NFWnRvir(c_sat, xyz[0], xyz[1], xyz[2], M_sat, Rvir_sat)
        #c_sat = sat_model[3]

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

    extract(kwargs)

    if 'sat2_model' in kwargs:
        if (sat2_model[0] == 'NFW'):
            c_sat2 = sat2_model[3]
            M_sat2 = sat2_model[1]
            Rvir_sat2 = sat2_model[2]
            A_host2 = a_NFWnRvir(c_sat2, xyz2[0], xyz2[1], xyz2[2], \
                                 M_sat2, Rvir_sat2)

        elif (sat2_model[0] == 'hernquist'):
            M_sat2 = sat2_model[1]
            rs_sat2 = sat2_model[2]
            A_host2 = a_hernquist(rs_sat2, xyz2[0], xyz2[1], xyz2[2],\
                                  M_sat2)

        elif (sat2_model[0] == 'plummer'):
            M_sat2 = sat2_model[1]
            rs_sat2 = sat2_model[2]
            A_host2 = a_plummer(rs_sat2, xyz2[0], xyz2[1], xyz2[2],\
                                M_sat2)

        Ax += A_host2[0]
        Ay += A_host2[1]
        Az += A_host2[2]

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


