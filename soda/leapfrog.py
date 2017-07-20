import numpy as np
from astropy import units, constants
from .acceleration import *
from .dynamical_friction import *

def LMC_models(model):
    """
    Return a LMC model:

    """

    lmc_pos = np.array([-1, -41, -28])
    lmc_vel = np.array([-57, -226, 221])

    # from
    # https://github.com/jngaravitoc/LMC-MW/blob/master/code/LMC_model.ipynb

    LMC_Ms = [3E10, 5E10, 8E10, 1E11, 1.8E11, 2.5E11]
    LMC_as = [3.0, 6.4, 10.4, 12.7, 20, 25.2]

    #LMC_as = [8.0, 11, 14, 15, 20, 22.5]

    if model == 'LMC1':
        LMC_model = ['hernquist', LMC_Ms[0], LMC_as[0] ,0]
        return lmc_pos, lmc_vel, LMC_model

    elif model == 'LMC2':
        LMC_model = ['hernquist', LMC_Ms[1], LMC_as[1] ,0]
        return lmc_pos, lmc_vel, LMC_model

    elif model == 'LMC3':
        LMC_model = ['hernquist', LMC_Ms[2], LMC_as[2] ,0]
        return lmc_pos, lmc_vel, LMC_model

    elif model == 'LMC4':
        LMC_model = ['hernquist', LMC_Ms[3], LMC_as[3] ,0]
        return lmc_pos, lmc_vel, LMC_model

    elif model == 'LMC5':
        LMC_model = ['hernquist', LMC_Ms[4], LMC_as[4] ,0]
        return lmc_pos, lmc_vel, LMC_model

    elif model == 'LMC6':
        LMC_model = ['hernquist', LMC_Ms[5], LMC_as[5] ,0]
        return lmc_pos, lmc_vel, LMC_model


def initialize_coordinates(n_points):
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)

    vx = np.zeros(n_points)
    vy = np.zeros(n_points)
    vz = np.zeros(n_points)

    ax = np.zeros(n_points)
    ay = np.zeros(n_points)
    az = np.zeros(n_points)

    return x, y, z, vx, vy, vz, ax, ay, az

def relative_coordinates(x1, y1, z1, x2, y2, z2, vx1, vy1, \
                         vz1, vx2, vy2, vz2):

    """
    compute relative coordinates.

    """

    xyz_rel = np.array([x1-x2, y1-y2, z1-z2])
    vxyz_rel = np.array([vx1-vx2, vy1-vy2, vz1-vz2])

    return xyz_rel, vxyz_rel

def extract(dct, namespace=None):
    # function that extracts variables from kwargs
    # from:
    # http://stackoverflow.com/questions/4357851/creating-or-assigning-variables-from-a-dictionary-in-python

    if not namespace: namespace = globals()
    namespace.update(dct)



def integrate_mw(time, pos_p, vel_p, \
                 host_model, direction=1, dt=0.01, ac=0, **kwargs):

    """
    Orbit integrator of a test particle around a MW-like halo at
    [0,0,0]

    Input:
    ------
    time: Time of the integration in Gyrs
    pos_p
    vel_p
    host_model: array('NFW'/'hernquist', Mass 1E10, Rvir/r_s, concentration)
    disk_params: array(Mass, a, b)
    bulge_params: array(Mass, r_s)
    ac (optional, default=0): No (0), Yes(1)
    direction (optional, default=1): Forward -1, Backwards=1
    dt: Time step for the integration (default dt=0.01 Gyrs)
    Output:
    ------

    t:
    pos_p
    vel_p

    TO-DO:
    ------
    0. Generalize to any MW model!!
    1. Generalize to N-test particles.
    2. Integrate with galpy/gala
    3. Used in arbitrary accelerations/SCF
    """

    extract(kwargs)


    conv_factor = 1.0227121650537077 # from km/s to Kpc/Gyr
    # h is the time step
    h = dt * direction
    n_points = int(time / dt) # Make this an input parameter!

    t = np.zeros(n_points)

    x_p, y_p, z_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p= initialize_coordinates(n_points)

    t[0] = 0 # Make this an input parameter?


    x_p[0] = pos_p[0]
    y_p[0] = pos_p[1]
    z_p[0] = pos_p[2]

    vx_p[0] = vel_p[0]*conv_factor
    vy_p[0] = vel_p[1]*conv_factor
    vz_p[0] = vel_p[2]*conv_factor

    # half step
    # Here I assume the host galaxy starts at position (0, 0, 0) and then its

    # initial v[1] is (0, 0, 0)
    t[1] = t[0] - h

    x_p[1] = x_p[0] - h * vx_p[0]
    y_p[1] = y_p[0] - h * vy_p[0]
    z_p[1] = z_p[0] - h * vz_p[0]

    vx_p[1] = vx_p[0] - h * ax_p[0]
    vy_p[1] = vy_p[0] - h * ay_p[0]
    vz_p[1] = vz_p[0] - h * az_p[0]


    if ('disk_params' and 'bulge_params') in kwargs:
        ax_p[1], ay_p[1], az_p[1] = acc_sat_helper([x_p[1],\
                                    y_p[1],z_p[1]], host_model, ac,\
                                    disk_params=disk_params, \
                                    bulge_params=bulge_params)


    ax_p[1], ay_p[1], az_p[1] = acc_sat_helper([x_p[1],\
                                y_p[1],z_p[1]], host_model, ac)


    for i in range(2, len(x_p)):
        t[i] = t[i-1] - h

        x_p[i] = x_p[i-2] - 2 * h * vx_p[i-1]
        y_p[i] = y_p[i-2] - 2 * h * vy_p[i-1]
        z_p[i] = z_p[i-2] - 2 * h * vz_p[i-1]

        vx_p[i] = vx_p[i-2] - 2 * h * ax_p[i-1]
        vy_p[i] = vy_p[i-2] - 2 * h * ay_p[i-1]
        vz_p[i] = vz_p[i-2] - 2 * h * az_p[i-1]

        if ('disk_params' and 'bulge_params') in kwargs:
            ax_p[i], ay_p[i], az_p[i] = acc_sat_helper([x_p[i],\
                                        y_p[i],z_p[i]], host_model, ac,\
                                        disk_params=disk_params, \
                                        bulge_params=bulge_params)

        ax_p[i], ay_p[i], az_p[i] = acc_sat_helper([x_p[i],\
                                    y_p[i],z_p[i]], host_model, ac)

    return t, np.array([x_p, y_p, z_p]).T, np.array([vx_p, vy_p, vz_p]).T/conv_factor


def integrate_sat(time, pos_host, vel_host, host_model, disk_params,\
                  bulge_params, ac=0, dfric=1, alpha=[0, 1], host_move=1,\
                  direction=1, dt=0.01, C=0, **kwargs):

## to do: generalize to any MW potential, maybe without a disk or

    """
    Orbit integrator of satellites around a host halo with disk and
    bulge. At the moment it can integrate one or two satellites around
    a host taking into account dynamical friction. It also can
    integrate a test particle in that system.

    Input:
    ------

    time: Time of the integration in Gyrs
    pos_host: array with the initial cartesian position of the host.
    vel_host: array with the initial cartesian velocity of the host.
    host_model: array('NFW'/'hernquist', Mass 1E10, Rvir/r_s, concentration)
    disk_params: array(Mass, a, b)
    bulge_params: array(Mass, r_s)
    ac (optional, default=0): No (0), Yes(1)
    dfric: Include dynamical friction No(0), default Yes(1)
    alpha: array(cl, alpha, L, C), cl=0 (), cl=1 (Van der Marel)
    host_move (optional, default=1): No(0), Yes(1)
    direction (optional, default=1): Forward -1, Backwards=1
    dt: Time step for the integration (default dt=0.01 Gyrs)

    kwargs : 

    satellite_model

    pos_sat

    vel_sat

    satellite_model2

    pos_sat2

    vel_sat2

    lmc_model : string
        octopus internally have models for the LMC. you can choose
        between: 'LMC1', 'LMC2', 'LMC3', 'LMC4', 'LMC5', 'LMC6' for 
        details of the models see LMC_models  

    pos_p : 

    vel_p : 

    alpha2 : dobule 
            A different df parameter for the satellite.
    Output:
    -------

    t : float
        Look back time of the orbit.
    pos_sat : array
    vel_sat : array 
    pos_host : array 
    vel_host : array 
    pos_p : array 
    vel_p : array 

    TO-DO:
    ------

    1. Generalize for N satellites.
    2. Integrate with galpy/gala
    3. Used in arbitrary accelerations/SCF

    """

    extract(kwargs)

    if 'alpha2' not in kwargs:
        alpha2=alpha

    if 'lmc_model' in kwargs:
        print('using the ', lmc_model)
        lmc_pos, lmc_vel, sat_model= LMC_models(lmc_model)
    else:
        sat_model = satellite_model

    if 'satellite_model2' in kwargs:
        sat_model2 = satellite_model2

    conv_factor = 1.0227121650537077 # from km/s to Kpc/Gyr

    # h is the time step
    h = dt * direction
    n_points = int(time / dt) # Make this an input parameter!

    t = np.zeros(n_points)

    x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc, vz_lmc, ax_lmc, ay_lmc, az_lmc = initialize_coordinates(n_points)
    x_mw, y_mw, z_mw, vx_mw, vy_mw, vz_mw, ax_mw, ay_mw, az_mw = initialize_coordinates(n_points)

    if 'pos_sat2' in kwargs:
         x_sag, y_sag, z_sag, vx_sag, vy_sag, vz_sag, ax_sag, ay_sag, az_sag = initialize_coordinates(n_points)

    t[0] = 0 # Make this an input parameter?


    if 'lmc_model' in kwargs:
        x_lmc[0] = lmc_pos[0]
        y_lmc[0] = lmc_pos[1]
        z_lmc[0] = lmc_pos[2]
        vx_lmc[0] = lmc_vel[0]*conv_factor
        vy_lmc[0] = lmc_vel[1]*conv_factor
        vz_lmc[0] = lmc_vel[2]*conv_factor

    if 'pos_sat' in kwargs:
        x_lmc[0] = pos_sat[0]
        y_lmc[0] = pos_sat[1]
        z_lmc[0] = pos_sat[2]
        vx_lmc[0] = vel_sat[0]*conv_factor
        vy_lmc[0] = vel_sat[1]*conv_factor
        vz_lmc[0] = vel_sat[2]*conv_factor

    if 'pos_sat2' in kwargs:
        x_sag[0] = pos_sat2[0]
        y_sag[0] = pos_sat2[1]
        z_sag[0] = pos_sat2[2]

        vx_sag[0] = vel_sat2[0]*conv_factor
        vy_sag[0] = vel_sat2[1]*conv_factor
        vz_sag[0] = vel_sat2[2]*conv_factor

    x_mw[0] = pos_host[0]
    y_mw[0] = pos_host[1]
    z_mw[0] = pos_host[2]

    vx_mw[0] = vel_host[0]*conv_factor
    vy_mw[0] = vel_host[1]*conv_factor
    vz_mw[0] = vel_host[2]*conv_factor


    ## Relative positions definitions:
    ## pos_hs relative position between the satellite and the host
    ## pos_hs2 relative position between the satellite and the host
    ## pos_ss: relative distance between the satellites


    pos_hs0, vel_hs0 = relative_coordinates(x_lmc[0], y_lmc[0], z_lmc[0], x_mw[0],\
                                            y_mw[0], z_mw[0], vx_lmc[0], \
                                            vy_lmc[0], vz_lmc[0], vx_mw[0], \
                                            vy_mw[0], vz_mw[0])


    print(C)
    ax_lmc[0], ay_lmc[0], az_lmc[0] = acc_sat(pos_hs0, vel_hs0, host_model, sat_model, \
                                              disk_params, bulge_params, ac, dfric, C ,alpha)

    ax_mw[0], ay_mw[0], az_mw[0] = acc_host(-pos_hs0, -vel_hs0, host_model, sat_model)

    if 'pos_sat2' in kwargs:

        pos_hs20, vel_hs20 = relative_coordinates(x_sag[0], y_sag[0], z_sag[0], x_mw[0],\
                                                  y_mw[0], z_mw[0], vx_sag[0], \
                                                  vy_sag[0], vz_sag[0], vx_mw[0], \
                                                  vy_mw[0], vz_mw[0])


        pos_ss0, vel_ss0 = relative_coordinates(x_sag[0], y_sag[0], z_sag[0], x_lmc[0],\
                                                y_lmc[0], z_lmc[0], vx_sag[0], \
                                                vy_sag[0], vz_sag[0], vx_lmc[0], \
                                                vy_lmc[0], vz_lmc[0])

        ax_lmc[0], ay_lmc[0], az_lmc[0] = acc_sat(pos_hs0, vel_hs0,\
                                                  host_model, \
                                                  sat_model,
                                                  disk_params,\
                                                  bulge_params,\
                                                  ac, dfric,\
                                                  C, alpha,\
                                                  xyz2=-pos_ss0,\
                                                  sat2_model=sat_model2)

        ax_sag[0], ay_sag[0], az_sag[0] = acc_sat(pos_hs20,\
                                                  vel_hs20,\
                                                  host_model,\
                                                  sat_model2,\
                                                  disk_params,\
                                                  bulge_params,\
                                                  ac, C, alpha2,\
                                                  xyz2=pos_ss0,\
                                                  sat2_model= sat_model)

        ax_mw[0], ay_mw[0], az_mw[0] = acc_host(-pos_hs0, -vel_hs0,\
                                                 host_model,\
                                                 sat_model,\
                                                 xyz2=-pos_hs20,\
                                                 sat2_model=sat_model2)

    # half step
    # Here I assume the host galaxy starts at position (0, 0, 0) and then its

    print('Host: ', x_mw[0], y_mw[0], z_mw[0])
    print('Satellite 1 :' , x_lmc[0], y_lmc[0], z_lmc[0])


    # initial v[1] is (0, 0, 0)
    t[1] = t[0] - h
    x_lmc[1] = x_lmc[0] - h * vx_lmc[0]
    y_lmc[1] = y_lmc[0] - h * vy_lmc[0]
    z_lmc[1] = z_lmc[0] - h * vz_lmc[0]

    vx_lmc[1] = vx_lmc[0] - h * ax_lmc[0]
    vy_lmc[1] = vy_lmc[0] - h * ay_lmc[0]
    vz_lmc[1] = vz_lmc[0] - h * az_lmc[0]

    pos_hs1, vel_hs1 = relative_coordinates(x_lmc[1], y_lmc[1], z_lmc[1], x_mw[1],\
                                            y_mw[1], z_mw[1], vx_lmc[1], \
                                            vy_lmc[1], vz_lmc[1], vx_mw[1], \
                                            vy_mw[1], vz_mw[1])


    ax_lmc[1], ay_lmc[1], az_lmc[1] = acc_sat(pos_hs1, vel_hs1,\
                                              host_model,\
                                              sat_model,disk_params,\
                                              bulge_params, ac,\
                                              dfric, C, alpha)
    if 'pos_sat2' in kwargs:

        print('Satellite 2: ', x_sag[0], y_sag[0], z_sag[0])
        x_sag[1] = x_sag[0] - h * vx_sag[0]
        y_sag[1] = y_sag[0] - h * vy_sag[0]
        z_sag[1] = z_sag[0] - h * vz_sag[0]

        vx_sag[1] = vx_sag[0] - h * ax_sag[0]
        vy_sag[1] = vy_sag[0] - h * ay_sag[0]
        vz_sag[1] = vz_sag[0] - h * az_sag[0]

        pos_hs21, vel_hs21 = relative_coordinates(x_sag[1], y_sag[1], z_sag[1], x_mw[1],\
                                                  y_mw[1], z_mw[1], vx_sag[1], \
                                                  vy_sag[1], vz_sag[1], vx_mw[1], \
                                                  vy_mw[1], vz_mw[1])


        pos_ss1, vel_ss1 = relative_coordinates(x_sag[1], y_sag[1], z_sag[1], x_lmc[1],\
                                                y_lmc[1], z_lmc[1], vx_sag[1], \
                                                vy_sag[1], vz_sag[1], vx_lmc[1], \
                                                vy_lmc[1], vz_lmc[1])

        ax_lmc[1], ay_lmc[1], az_lmc[1] = acc_sat(pos_hs1, vel_hs1,\
                                                 host_model,\
                                                 sat_model, disk_params,\
                                                 bulge_params, ac,\
                                                 dfric, alpha, xyz2=-pos_ss1, sat2_model = sat_model2)

        ax_sag[1], ay_sag[1], az_sag[1] = acc_sat(pos_hs21, vel_hs21, host_model, sat_model2,\
                                                  disk_params, bulge_params, ac, dfric,\
                                                  alpha2, xyz2=pos_ss1, sat2_model= sat_model)

    if (host_move==1):
        x_mw[1] = x_mw[0] - h * vx_mw[0]
        y_mw[1] = y_mw[0] - h * vy_mw[0]
        z_mw[1] = z_mw[0] - h * vz_mw[0]

        vx_mw[1] = vx_mw[0] - h * ax_mw[0]
        vy_mw[1] = vy_mw[0] - h * ay_mw[0]
        vz_mw[1] = vz_mw[0] - h * az_mw[0]

        pos_hs1, vel_hs1 = relative_coordinates(x_lmc[1], y_lmc[1], z_lmc[1], x_mw[1],\
                                                y_mw[1], z_mw[1], vx_lmc[1], \
                                                vy_lmc[1], vz_lmc[1], vx_mw[1], \
                                                vy_mw[1], vz_mw[1])

        ax_mw[1], ay_mw[1], az_mw[1] = acc_host(-pos_hs1, -vel_hs1, host_model, sat_model)

        ax_lmc[1], ay_lmc[1], az_lmc[1] = acc_sat(pos_hs1, vel_hs1,\
                                                 host_model,\
                                                 sat_model,disk_params,\
                                                 bulge_params, ac,\
                                                 dfric, C, alpha)

        if 'pos_sat2' in kwargs:


            pos_hs21, vel_hs21 = relative_coordinates(x_sag[1], y_sag[1], z_sag[1], x_mw[1],\
                                                      y_mw[1], z_mw[1], vx_sag[1], \
                                                      vy_sag[1], vz_sag[1], vx_mw[1], \
                                                      vy_mw[1], vz_mw[1])

            ax_lmc[1], ay_lmc[1], az_lmc[1] = acc_sat(pos_hs1, vel_hs1, host_model, sat_model,\
                                                      disk_params, bulge_params, ac, dfric, C,\
                                                      alpha, xyz2=-pos_ss1, sat2_model= sat_model2)

            ax_sag[1], ay_sag[1], az_sag[1] = acc_sat(pos_hs21, vel_hs21, host_model, sat_model2,\
                                                      disk_params, bulge_params, ac, dfric, C,\
                                                      alpha2, xyz2=pos_ss1, sat2_model= sat_model)

            ax_mw[1], ay_mw[1], az_mw[1] = acc_host(-pos_hs1, -vel_hs1, host_model, sat_model,\
                                                     xyz2=-pos_hs21, sat2_model=sat_model2)





    for i in range(2, len(x_lmc)):
        t[i] = t[i-1] - h
        x_lmc[i] = x_lmc[i-2] - 2 * h * vx_lmc[i-1]
        y_lmc[i] = y_lmc[i-2] - 2 * h * vy_lmc[i-1]
        z_lmc[i] = z_lmc[i-2] - 2 * h * vz_lmc[i-1]

        vx_lmc[i] = vx_lmc[i-2] - 2 * h * ax_lmc[i-1]
        vy_lmc[i] = vy_lmc[i-2] - 2 * h * ay_lmc[i-1]
        vz_lmc[i] = vz_lmc[i-2] - 2 * h * az_lmc[i-1]


        pos_hsi, vel_hsi = relative_coordinates(x_lmc[i], y_lmc[i], z_lmc[i], x_mw[i],\
                                                y_mw[i], z_mw[i], vx_lmc[i], \
                                                vy_lmc[i], vz_lmc[i], vx_mw[i], \
                                                vy_mw[i], vz_mw[i])



        ax_lmc[i], ay_lmc[i], az_lmc[i] = acc_sat(pos_hsi, vel_hsi, host_model, sat_model,\
                                                  disk_params, bulge_params, ac, dfric, C,\
                                                  alpha)
        if 'pos_sat2' in kwargs:

            x_sag[i] = x_sag[i-2] - 2 * h * vx_sag[i-1]
            y_sag[i] = y_sag[i-2] - 2 * h * vy_sag[i-1]
            z_sag[i] = z_sag[i-2] - 2 * h * vz_sag[i-1]

            vx_sag[i] = vx_sag[i-2] - 2 * h * ax_sag[i-1]
            vy_sag[i] = vy_sag[i-2] - 2 * h * ay_sag[i-1]
            vz_sag[i] = vz_sag[i-2] - 2 * h * az_sag[i-1]

            pos_hs2i, vel_hs2i = relative_coordinates(x_sag[i], y_sag[i], z_sag[i], x_mw[i],\
                                                      y_mw[i], z_mw[i], vx_sag[i], \
                                                      vy_sag[i], vz_sag[i], vx_mw[i], \
                                                      vy_mw[i], vz_mw[i])


            pos_ssi, vel_ssi = relative_coordinates(x_sag[i], y_sag[i], z_sag[i], x_lmc[i],\
                                                    y_lmc[i], z_lmc[i], vx_sag[i], \
                                                    vy_sag[i], vz_sag[i], vx_lmc[i], \
                                                    vy_lmc[i], vz_lmc[i])

            ax_lmc[i], ay_lmc[i], az_lmc[i] = acc_sat(pos_hsi, vel_hsi, host_model, sat_model,\
                                                      disk_params, bulge_params, ac, dfric, C,\
                                                      alpha, xyz2=-pos_ssi, sat2_model=sat_model2)

            ax_sag[i], ay_sag[i], az_sag[i] = acc_sat(pos_hs2i, vel_hs2i, host_model, sat_model2,\
                                                      disk_params, bulge_params, ac, dfric, C,\
                                                      alpha2, xyz2=pos_ssi, sat2_model=sat_model)
        if (host_move==1):
            x_mw[i] = x_mw[i-2] - 2 * h * vx_mw[i-1]
            y_mw[i] = y_mw[i-2] - 2 * h * vy_mw[i-1]
            z_mw[i] = z_mw[i-2] - 2 * h * vz_mw[i-1]

            vx_mw[i] = vx_mw[i-2] - 2 * h * ax_mw[i-1]
            vy_mw[i] = vy_mw[i-2] - 2 * h * ay_mw[i-1]
            vz_mw[i] = vz_mw[i-2] - 2 * h * az_mw[i-1]


            pos_hsi, vel_hsi= relative_coordinates(x_lmc[i], y_lmc[i], z_lmc[i], x_mw[i],\
                                                   y_mw[i], z_mw[i], vx_lmc[i], \
                                                   vy_lmc[i], vz_lmc[i], vx_mw[i], \
                                                   vy_mw[i], vz_mw[i])

            ax_mw[i], ay_mw[i], az_mw[i] = acc_host(-pos_hsi, -vel_hsi, host_model, sat_model)

            ax_lmc[i], ay_lmc[i], az_lmc[i] = acc_sat(pos_hsi, vel_hsi, host_model, sat_model,\
                                                      disk_params, bulge_params, ac, dfric, C,\
                                                      alpha)

            if 'pos_sat2' in kwargs:


                pos_hs2i, vel_hs2i = relative_coordinates(x_sag[i], y_sag[i], z_sag[i], x_mw[i],\
                                                          y_mw[i], z_mw[i], vx_sag[i], \
                                                          vy_sag[i], vz_sag[i], vx_mw[i], \
                                                          vy_mw[i], vz_mw[i])


                ax_lmc[i], ay_lmc[i], az_lmc[i] = acc_sat(pos_hsi, vel_hsi, host_model, sat_model,\
                                                          disk_params, bulge_params, ac, dfric,\
                                                          C,alpha, xyz2=-pos_ssi, sat2_model=sat_model2)

                ax_sag[i], ay_sag[i], az_sag[i] = acc_sat(pos_hs2i, vel_hs2i, host_model, sat_model2,\
                                                          disk_params, bulge_params, ac, dfric,\
                                                          C, alpha2,\
                                                          xyz2=pos_ssi,\
                                                          sat2_model=sat_model)

                ax_mw[i], ay_mw[i], az_mw[i] = acc_host(-pos_hsi, -vel_hsi, host_model, sat_model,\
                                                        xyz2=-pos_hs2i, sat2_model=sat_model2)



    if 'pos_p' in kwargs:
        x_p, y_p, z_p, vx_p, vy_p, vz_p = integrate_sat_helper(time,\
                                                               n_points,\
                                                               x_mw,\
                                                               y_mw,\
                                                               z_mw,\
                                                               vx_mw,\
                                                               vy_mw,\
                                                               vz_mw,\
                                                               x_lmc,\
                                                               y_lmc,\
                                                               z_lmc,\
                                                               vx_lmc,\
                                                               vy_lmc,\
                                                               vz_lmc,\
                                                               sat_model,\
                                                               host_model,\
                                                               disk_params,\
                                                               bulge_params,\
                                                               pos_p,\
                                                               vel_p,\
                                                               ac, dt,\
                                                               direction)

        if 'pos_sat2' in kwargs:
            x_p, y_p, z_p, vx_p, vy_p, vz_p = integrate_sat_helper(time,\
                                                                   n_points,\
                                                                   x_mw,\
                                                                   y_mw,\
                                                                   z_mw,\
                                                                   vx_mw,\
                                                                   vy_mw,\
                                                                   vz_mw,\
                                                                   x_lmc,\
                                                                   y_lmc,\
                                                                   z_lmc,\
                                                                   vx_lmc,\
                                                                   vy_lmc,\
                                                                   vz_lmc,\
                                                                   sat_model,\
                                                                   host_model,\
                                                                   disk_params, \
                                                                   bulge_params,\
                                                                   pos_p,\
                                                                   vel_p,\
                                                                   ac,\
                                                                   dt,\
                                                                   direction,\
                                                                   x_sag=x_sag,\
                                                                   y_sag=y_sag,\
                                                                   z_sag=z_sag,\
                                                                   vx_sag=vx_sag,\
                                                                   vy_sag=vy_sag,\
                                                                   vz_sag=vz_sag,\
                                                                   sat_model2=sat_model2)

            del(sat_model2)
        if 'pos_sat2' in kwargs:

            return t, np.array([x_lmc, y_lmc, z_lmc]).T,\
                   np.array([vx_lmc, vy_lmc, vz_lmc]).T/conv_factor,\
                   np.array([x_mw, y_mw, z_mw]).T,\
                   np.array([vx_mw,vy_mw, vz_mw]).T/conv_factor,\
                   np.array([x_p, y_p, z_p]).T,\
                   np.array([vx_p, vy_p,vz_p]).T/conv_factor,\
                   np.array([x_sag, y_sag, z_sag]).T,\
                   np.array([vx_sag, vy_sag,vz_sag]).T/conv_factor

        else:
            return t, np.array([x_lmc, y_lmc, z_lmc]).T,\
                   np.array([vx_lmc, vy_lmc, vz_lmc]).T/conv_factor,\
                   np.array([x_mw, y_mw, z_mw]).T,\
                   np.array([vx_mw, vy_mw, vz_mw]).T/conv_factor,\
                   np.array([x_p, y_p, z_p]).T,\
                   np.array([vx_p, vy_p, vz_p]).T/conv_factor


    else:

        return t, np.array([x_lmc, y_lmc, z_lmc]).T,\
               np.array([vx_lmc, vy_lmc, vz_lmc]).T/conv_factor, \
               np.array([x_mw, y_mw, z_mw]).T, \
               np.array([vx_mw,vy_mw,vz_mw]).T/conv_factor


def integrate_sat_helper(time, n_points, x_mw, y_mw, z_mw, vx_mw, vy_mw,\
                         vz_mw, x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc,\
                         vz_lmc, sat_model, host_model, disk_params, \
                         bulge_params, pos_sat, vel_sat, ac, dt, direction,\
                         **kwargs):

    extract(kwargs)
    conv_factor = 1.0227121650537077 # from km/s to Kpc/Gyr

    h = dt * direction
    n_points = int(time / dt) # Make this an input parameter!
    t = np.zeros(n_points)
    t[0]=0
    x_p, y_p, z_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p= initialize_coordinates(n_points)

    x_p[0] = pos_sat[0]
    y_p[0] = pos_sat[1]
    z_p[0] = pos_sat[2]

    vx_p[0] = vel_sat[0]*conv_factor
    vy_p[0] = vel_sat[1]*conv_factor
    vz_p[0] = vel_sat[2]*conv_factor

    pos_p2lmc_0, vel_p2lmc_0 = relative_coordinates(x_p[0],\
                                                    y_p[0],\
                                                    z_p[0],\
                                                    x_lmc[0],\
                                                    y_lmc[0],\
                                                    z_lmc[0],\
                                                    vx_p[0],\
                                                    vy_p[0],\
                                                    vz_p[0],\
                                                    vx_lmc[0],\
                                                    vy_lmc[0],\
                                                    vz_lmc[0])

    pos_p2mw_0, vel_p2mw_0 = relative_coordinates(x_p[0], y_p[0], z_p[0],\
                               x_mw[0], y_mw[0], z_mw[0], vx_p[0], vy_p[0],\
                               vz_p[0], vx_mw[0], vy_mw[0], vz_mw[0])

    if 'pos_sat2' in kwargs:

        pos_p2sag_0, vel_p2sag_0 = relative_coordinates(x_p[0], y_p[0], z_p[0],\
                                  x_sag[0], y_sag[0], z_sag[0], vx_p[0], vy_p[0],\
                                  vz_p[0], vx_sag[0], vy_sag[0], vz_sag[0])

    ax_p[0], ay_p[0], az_p[0] = particle_acceleration_LMC(pos_p2lmc_0, \
                                                 pos_p2mw_0, sat_model, host_model,\
                                                 disk_params, bulge_params,\
                                                 ac)

    if 'pos_sat2' in kwargs:

        ax_p[0], ay_p[0], az_p[0] = particle_acceleration_LMC(pos_p2lmc_0, \
                                                    pos_p2mw_0, sat_model, host_model,\
                                                    disk_params, bulge_params,\
                                                    ac, pos_sat2 = pos_p2sag_0,\
                                                    sat_model2=sat_model2)


    t[1] = t[0] - h
    x_p[1] = x_p[0] - h * vx_p[0]
    y_p[1] = y_p[0] - h * vy_p[0]
    z_p[1] = z_p[0] - h * vz_p[0]

    vx_p[1] = vx_p[0] - h * ax_p[0]
    vy_p[1] = vy_p[0] - h * ay_p[0]
    vz_p[1] = vz_p[0] - h * az_p[0]

    pos_p2lmc_1, vel_p2lmc_1 = relative_coordinates(x_p[1], y_p[1], z_p[1],\
                               x_lmc[1], y_lmc[1], z_lmc[1], vx_p[1], vy_p[1],\
                               vz_p[1], vx_lmc[1], vy_lmc[1], vz_lmc[1])

    pos_p2mw_1, vel_p2mw_1 = relative_coordinates(x_p[1], y_p[1], z_p[1],\
                               x_mw[1], y_mw[1], z_mw[1], vx_p[1], vy_p[1],\
                               vz_p[1], vx_mw[1], vy_mw[1], vz_mw[1])


    if 'pos_sat2' in kwargs:

        pos_p2sag_1, vel_p2sag_1 = relative_coordinates(x_p[1], y_p[1], z_p[1],\
                                  x_sag[1], y_sag[1], z_sag[1], vx_p[1], vy_p[1],\
                                  vz_p[1], vx_sag[1], vy_sag[1], vz_sag[1])

    ax_p[1], ay_p[1], az_p[1] = particle_acceleration_LMC(pos_p2lmc_1, \
                                                 pos_p2mw_1, sat_model, host_model,\
                                                 disk_params, bulge_params,\
                                                 ac)

    if 'pos_sat2' in kwargs:

        ax_p[1], ay_p[1], az_p[1] = particle_acceleration_LMC(pos_p2lmc_1, \
                                                    pos_p2mw_1, sat_model, host_model,\
                                                    disk_params, bulge_params,\
                                                     ac, pos_sat2 =pos_p2sag_1,\
                                                    sat_model2=sat_model2)


    for i in range(2, len(x_lmc)):
        t[i] = t[i-1] - h
        x_p[i] = x_p[i-2] - 2 * h * vx_p[i-1]
        y_p[i] = y_p[i-2] - 2 * h * vy_p[i-1]
        z_p[i] = z_p[i-2] - 2 * h * vz_p[i-1]

        vx_p[i] = vx_p[i-2] - 2 * h * ax_p[i-1]
        vy_p[i] = vy_p[i-2] - 2 * h * ay_p[i-1]
        vz_p[i] = vz_p[i-2] - 2 * h * az_p[i-1]

        pos_p2lmc_i, vel_p2lmc_i = relative_coordinates(x_p[i],\
                                                        y_p[i],\
                                                        z_p[i],\
                                                        x_lmc[i],\
                                                        y_lmc[i],\
                                                        z_lmc[i],\
                                                        vx_p[i],\
                                                        vy_p[i],\
                                                        vz_p[i],\
                                                        vx_lmc[i],\
                                                        vy_lmc[i],\
                                                        vz_lmc[i])

        pos_p2mw_i, vel_p2mw_i = relative_coordinates(x_p[i],\
                                                      y_p[i],\
                                                      z_p[i],\
                                                      x_mw[i],\
                                                      y_mw[i],\
                                                      z_mw[i],\
                                                      vx_p[i],\
                                                      vy_p[i],\
                                                      vz_p[i],\
                                                      vx_mw[i],\
                                                      vy_mw[i],\
                                                      vz_mw[i])

        if 'pos_sat2' in kwargs:

            pos_p2sag_i, vel_p2sag_i = relative_coordinates(x_p[i],\
                                                            y_p[i],\
                                                            z_p[i],\
                                                            x_sag[i],\
                                                            y_sag[i],\
                                                            z_sag[i],\
                                                            vx_p[i],\
                                                            vy_p[i],\
                                                            vz_p[i],\
                                                            vx_sag[i],\
                                                            vy_sag[i],\
                                                            vz_sag[i])


        ax_p[i], ay_p[i], az_p[i] = particle_acceleration_LMC(pos_p2lmc_i,\
                                                              pos_p2mw_i,\
                                                              sat_model,\
                                                              host_model,\
                                                              disk_params,\
                                                              bulge_params,\
                                                              ac)

        if 'pos_sat2' in kwargs:

            ax_p[i], ay_p[i], az_p[i] = particle_acceleration_LMC(pos_p2lmc_i,\
                                                       pos_p2mw_i, sat_model, host_model,\
                                                       disk_params, bulge_params,\
                                                       ac, pos_sat2 =pos_p2sag_i,\
                                                       sat_model2=sat_model2)

    return x_p, y_p, z_p, vx_p/conv_factor, vy_p/conv_factor, vz_p/conv_factor
