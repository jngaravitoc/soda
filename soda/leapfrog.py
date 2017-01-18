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
    # function that extracts variables
    # from:
    # http://stackoverflow.com/questions/4357851/creating-or-assigning-variables-from-a-dictionary-in-python

    if not namespace: namespace = globals()
    namespace.update(dct)



def integrate_mw(time, pos_p, vel_p, host_model, disk_params,\
                 bulge_params, ac=0, direction=1, dt=0.01, **kwargs):

# kwargs: pos_p, vel_p

## to do: generalize to any MW potential, maybe without a disk or
## with!

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
    1. Generalize to N- test particles.
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



    ax_p[0], ay_p[0], az_p[0] = acc_sat_helper([x_p[0],\
                                y_p[0],z_p[0]], host_model,\
                                disk_params, bulge_params, ac)

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



    ax_p[1], ay_p[1], az_p[1] = acc_sat_helper([x_p[1],\
                                y_p[1],z_p[1]], host_model,\
                                disk_params, bulge_params, ac)

    for i in range(2, len(x_p)):
        t[i] = t[i-1] - h

        x_p[i] = x_p[i-2] - 2 * h * vx_p[i-1]
        y_p[i] = y_p[i-2] - 2 * h * vy_p[i-1]
        z_p[i] = z_p[i-2] - 2 * h * vz_p[i-1]

        vx_p[i] = vx_p[i-2] - 2 * h * ax_p[i-1]
        vy_p[i] = vy_p[i-2] - 2 * h * ay_p[i-1]
        vz_p[i] = vz_p[i-2] - 2 * h * az_p[i-1]

        ax_p[i], ay_p[i], az_p[i] = acc_sat_helper([x_p[i],\
                                    y_p[i],z_p[i]], host_model,\
                                    disk_params, bulge_params, ac)


    return t, np.array([x_p, y_p, z_p]).T, np.array([vx_p, vy_p, vz_p]).T/conv_factor


def integrate_sat(time, pos_host, vel_host, host_model, disk_params,\
              bulge_params, ac=0, dfric=1, alpha=[0, 1], host_move=1, 
              direction=1, dt=0.01, **kwargs):

# kwargs: sat_model, pos_sat, vel_sat, pos_p, vel_p

## to do: generalize to any MW potential, maybe without a disk or
## with!

    """
    Orbit integrator:

    Input:
    ------
    time: Time of the integration in Gyrs
    pos_sat: array with the initial cartesian position of the satellite.
    vel_sat: array with the initial cartesian velocity of the satellite.
    pos_host: array with the initial cartesian position of the host.
    vel_host: array with the initial cartesian velocity of the host.
    host_model: array('NFW'/'hernquist', Mass 1E10, Rvir/r_s, concentration)
    sat_model: array('NFW'/'hernquist'/'plummer', Mass 1E10, Rvir/rs, concentration)
    disk_params: array(Mass, a, b)
    bulge_params: array(Mass, r_s)
    pos_p
    vel_p
    ac (optional, default=0): No (0), Yes(1)
    dfric: Include dynamical friction No(0), default Yes(1)
    alpha: array(cl, alpha, L, C), cl=0 (), cl=1 (Van der Marel)
    host_move (optional, default=1): No(0), Yes(1)
    direction (optional, default=1): Forward -1, Backwards=1
    dt: Time step for the integration (default dt=0.01 Gyrs)
    Output:
    ------

    t:
    pos_sat:
    vel_sat:
    pos_host:
    vel_host:
    pos_p
    vel_p

    TO-DO:
    ------

    1. Generalize for N satellites.
    2. Integrate with galpy/gala
    3. Used in arbitrary accelerations/SCF
    """

    extract(kwargs)

    print(vel_sat)
    print(sat_model)

    if 'lmc_model' in kwargs:
        print('using the ', lmc_model)
        lmc_pos, lmc_vel, sat_model= LMC_models(lmc_model)

    conv_factor = 1.0227121650537077 # from km/s to Kpc/Gyr
    # h is the time step
    h = dt * direction
    n_points = int(time / dt) # Make this an input parameter!

    t = np.zeros(n_points)

    x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc, vz_lmc, ax_lmc, ay_lmc, az_lmc = initialize_coordinates(n_points)
    # alternatively I could just copy the arrays, what's more
    # efficient?
    x_mw, y_mw, z_mw, vx_mw, vy_mw, vz_mw, ax_mw, ay_mw, az_mw = initialize_coordinates(n_points)

    t[0] = 0 # Make this an input parameter?


    if ((pos_sat == 'LMC') & (vel_sat== 'LMC')):
        x_lmc[0] = lmc_pos[0]
        y_lmc[0] = lmc_pos[1]
        z_lmc[0] = lmc_pos[2]
        vx_lmc[0] = lmc_vel[0]*conv_factor
        vy_lmc[0] = lmc_vel[1]*conv_factor
        vz_lmc[0] = lmc_vel[2]*conv_factor

    else:
        x_lmc[0] = pos_sat[0]
        y_lmc[0] = pos_sat[1]
        z_lmc[0] = pos_sat[2]
        vx_lmc[0] = vel_sat[0]*conv_factor
        vy_lmc[0] = vel_sat[1]*conv_factor
        vz_lmc[0] = vel_sat[2]*conv_factor

    x_mw[0] = pos_host[0]
    y_mw[0] = pos_host[1]
    z_mw[0] = pos_host[2]

    vx_mw[0] = vel_host[0]*conv_factor
    vy_mw[0] = vel_host[1]*conv_factor
    vz_mw[0] = vel_host[2]*conv_factor


    pos_0, vel_0 = relative_coordinates(x_lmc[0], y_lmc[0], z_lmc[0], x_mw[0],\
                                      y_mw[0], z_mw[0], vx_lmc[0], \
                                      vy_lmc[0], vz_lmc[0], vx_mw[0], \
                                      vy_mw[0], vz_mw[0])

    ax_lmc[0] = acc_sat(pos_0, vel_0, host_model, sat_model, \
                        disk_params, bulge_params, ac, dfric, alpha)[0]

    ay_lmc[0] = acc_sat(pos_0, vel_0, host_model, sat_model, \
                        disk_params, bulge_params, ac, dfric, alpha)[1]

    az_lmc[0] = acc_sat(pos_0, vel_0, host_model, sat_model, \
                    disk_params, bulge_params, ac, dfric, alpha)[2]

    ax_mw[0] = acc_host(-pos_0, -vel_0, host_model, sat_model)[0]
    ay_mw[0] = acc_host(-pos_0, -vel_0, host_model, sat_model)[1]
    az_mw[0] = acc_host(-pos_0, -vel_0, host_model, sat_model)[2]

    # half step
    # Here I assume the host galaxy starts at position (0, 0, 0) and then its

    # initial v[1] is (0, 0, 0)
    t[1] = t[0] - h
    x_lmc[1] = x_lmc[0] - h * vx_lmc[0]
    y_lmc[1] = y_lmc[0] - h * vy_lmc[0]
    z_lmc[1] = z_lmc[0] - h * vz_lmc[0]

    vx_lmc[1] = vx_lmc[0] - h * ax_lmc[0]
    vy_lmc[1] = vy_lmc[0] - h * ay_lmc[0]
    vz_lmc[1] = vz_lmc[0] - h * az_lmc[0]


    pos_1, vel_1 = relative_coordinates(x_lmc[1], y_lmc[1], z_lmc[1], x_mw[1],\
                                      y_mw[1], z_mw[1], vx_lmc[1], \
                                      vy_lmc[1], vz_lmc[1], vx_mw[1], \
                                      vy_mw[1], vz_mw[1])

    if (host_move==1):
        x_mw[1] = x_mw[0] - h * vx_mw[0]
        y_mw[1] = y_mw[0] - h * vy_mw[0]
        z_mw[1] = z_mw[0] - h * vz_mw[0]

        vx_mw[1] = vx_mw[0] - h * ax_mw[0]
        vy_mw[1] = vy_mw[0] - h * ay_mw[0]
        vz_mw[1] = vz_mw[0] - h * az_mw[0]

        pos_1, vel_1 = relative_coordinates(x_lmc[1], y_lmc[1], z_lmc[1], x_mw[1],\
                                          y_mw[1], z_mw[1], vx_lmc[1], \
                                          vy_lmc[1], vz_lmc[1], vx_mw[1], \
                                          vy_mw[1], vz_mw[1])

        ax_mw[1] = acc_host(-pos_1, -vel_1, host_model, sat_model)[0]
        ay_mw[1] = acc_host(-pos_1, -vel_1, host_model, sat_model)[1]
        az_mw[1] = acc_host(-pos_1, -vel_1, host_model, sat_model)[2]


    ax_lmc[1] = acc_sat(pos_1, vel_1, host_model, sat_model\
                   ,disk_params, bulge_params, ac, dfric, alpha)[0]
    ay_lmc[1] = acc_sat(pos_1, vel_1, host_model, sat_model\
                   ,disk_params, bulge_params, ac, dfric, alpha)[1]
    az_lmc[1] = acc_sat(pos_1, vel_1, host_model, sat_model\
                   ,disk_params, bulge_params, ac, dfric, alpha)[2]

    for i in range(2, len(x_lmc)):
        t[i] = t[i-1] - h
        x_lmc[i] = x_lmc[i-2] - 2 * h * vx_lmc[i-1]
        y_lmc[i] = y_lmc[i-2] - 2 * h * vy_lmc[i-1]
        z_lmc[i] = z_lmc[i-2] - 2 * h * vz_lmc[i-1]

        vx_lmc[i] = vx_lmc[i-2] - 2 * h * ax_lmc[i-1]
        vy_lmc[i] = vy_lmc[i-2] - 2 * h * ay_lmc[i-1]
        vz_lmc[i] = vz_lmc[i-2] - 2 * h * az_lmc[i-1]

        pos_i, vel_i = relative_coordinates(x_lmc[i], y_lmc[i], z_lmc[i], x_mw[i],\
                                          y_mw[i], z_mw[i], vx_lmc[i], \
                                          vy_lmc[i], vz_lmc[i], vx_mw[i], \
                                          vy_mw[i], vz_mw[i])


        if (host_move==1):
            x_mw[i] = x_mw[i-2] - 2 * h * vx_mw[i-1]
            y_mw[i] = y_mw[i-2] - 2 * h * vy_mw[i-1]
            z_mw[i] = z_mw[i-2] - 2 * h * vz_mw[i-1]

            vx_mw[i] = vx_mw[i-2] - 2 * h * ax_mw[i-1]
            vy_mw[i] = vy_mw[i-2] - 2 * h * ay_mw[i-1]
            vz_mw[i] = vz_mw[i-2] - 2 * h * az_mw[i-1]


            pos_i, vel_i = relative_coordinates(x_lmc[i], y_lmc[i],z_lmc[i], x_mw[i],\
                                              y_mw[i], z_mw[i], vx_lmc[i], \
                                              vy_lmc[i], vz_lmc[i], vx_mw[i], \
                                              vy_mw[i], vz_mw[i])

            ax_mw[i] = acc_host(-pos_i, -vel_i, host_model, sat_model)[0]
            ay_mw[i] = acc_host(-pos_i, -vel_i, host_model, sat_model)[1]
            az_mw[i] = acc_host(-pos_i, -vel_i, host_model, sat_model)[2]



        ax_lmc[i] = acc_sat(pos_i, vel_i, host_model, sat_model\
                       ,disk_params, bulge_params, ac, dfric, alpha)[0]
        ay_lmc[i] = acc_sat(pos_i, vel_i, host_model, sat_model\
                       ,disk_params, bulge_params, ac, dfric, alpha)[1]
        az_lmc[i] = acc_sat(pos_i, vel_i, host_model, sat_model\
                       ,disk_params, bulge_params, ac, dfric, alpha)[2]


    if pos_p == True:
        x_p, y_p, z_p, vx_p, vy_p, vz_p = integrate_sat_helper(n_points, x_mw, y_mw, z_mw, vx_mw, vy_mw,\
                                          vz_mw, x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc,\
                                          vz_lmc, sat_model, host_model, disk_params, \
                                          bulge_params, ac)

    return t, np.array([x_lmc, y_lmc, z_lmc]).T, np.array([vx_lmc, vy_lmc, vz_lmc]).T/conv_factor, \
           np.array([x_mw, y_mw, z_mw]).T, np.array([vx_mw,vy_mw,vz_mw]).T/conv_factor,\
           np.array([x_p, y_p, z_p]).T, np.array([vx_p, vy_p, vz_p]).T/conv_factor


def integrate_sat_helper(n_points, x_mw, y_mw, z_mw, vx_mw, vy_mw,\
                         vz_mw, x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc,\
                         vz_lmc, sat_model, host_model, disk_params, \
                         bulge_params, ac):

    conv_factor = 1.0227121650537077 # from km/s to Kpc/Gyr

    h = dt * direction
    n_points = int(time / dt) # Make this an input parameter!

    t[0]=0
    x_p, y_p, z_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p= initialize_coordinates(n_points)

    x_p[0] = pos_p[0]
    y_p[0] = pos_p[1]
    z_p[0] = pos_p[2]

    vx_p[0] = vel_p[0]*conv_factor
    vy_p[0] = vel_p[1]*conv_factor
    vz_p[0] = vel_p[2]*conv_factor


    pos_p2lmc_0, vel_p2lmc_0 = relative_coordinates(x_p[0], y_p[0], z_p[0],\
                               x_lmc[0], y_lmc[0], z_lmc[0],vx_p[0], vy_p[0],\
                               vz_p[0], vx_lmc[0], vy_lmc[0], vz_lmc[0])

    pos_p2mw_0, vel_p2mw_0 = relative_coordinates(x_p[0], y_p[0], z_p[0],\
                               x_mw[0], y_mw[0], z_mw[0],vx_p[0], vy_p[0],\
                               vz_p[0], vx_mw[0], vy_mw[0], vz_mw[0])

    ax_p[0], ay_p[0], az_p[0] = particle_acceleartion_LMC(pos_p2lmc_0, \
                                                 pos_p2mw_0, sat_model, host_model,\
                                                 disk_params, bulge_params,\
                                                 ac)


    t[1] = t[0] - h
    x_p[1] = x_p[0] - h * vx_p[0]
    y_p[1] = y_p[0] - h * vy_p[0]
    z_p[1] = z_p[0] - h * vz_p[0]

    vx_p[1] = vx_p[0] - h * ax_p[0]
    vy_p[1] = vy_p[0] - h * ay_p[0]
    vz_p[1] = vz_p[0] - h * az_p[0]

    pos_p2lmc_1, vel_p2lmc_1 = relative_coordinates(x_p[1], y_p[1], z_p[1],\
                               x_lmc[1], y_lmc[1], z_lmc[1],vx_p[1], vy_p[1],\
                               vz_p[1], vx_lmc[1], vy_lmc[1], vz_lmc[1])

    pos_p2mw_1, vel_p2mw_1 = relative_coordinates(x_p[1], y_p[1], z_p[1],\
                               x_mw[1], y_mw[1], z_mw[1],vx_p[1], vy_p[1],\
                               vz_p[1], vx_mw[1], vy_mw[1], vz_mw[1])



    ax_p[1], ay_p[1], az_p[1] = particle_acceleartion_LMC(pos_p2lmc_1, \
                                                 pos_p2mw_1, sat_model, host_model,\
                                                 disk_params, bulge_params,\
                                                 ac)



    for i in range(2, len(x_lmc)):
        t[i] = t[i-1] - h
        x_p[i] = x_p[i-2] - 2 * h * vx_p[i-1]
        y_p[i] = y_p[i-2] - 2 * h * vy_p[i-1]
        z_p[i] = z_p[i-2] - 2 * h * vz_p[i-1]

        vx_p[i] = vx_p[i-2] - 2 * h * ax_p[i-1]
        vy_p[i] = vy_p[i-2] - 2 * h * ay_p[i-1]
        vz_p[i] = vz_p[i-2] - 2 * h * az_p[i-1]

        pos_p2lmc_i, vel_p2lmc_i = relative_coordinates(x_p[i],y_p[i], z_p[i],\
                                   x_lmc[i], y_lmc[i], z_lmc[i],vx_p[i], vy_p[i],\
                                   vz_p[i], vx_lmc[i], vy_lmc[i], vz_lmc[i])

        pos_p2mw_i, vel_p2mw_i = relative_coordinates(x_p[i], y_p[i], z_p[i],\
                                  x_mw[i], y_mw[i], z_mw[i],vx_p[i], vy_p[i],\
                                  vz_p[i], vx_mw[i], vy_mw[i], vz_mw[i])


        ax_p[i], ay_p[i], az_p[i] = particle_acceleartion_LMC(pos_p2lmc_i, \
                                                 pos_p2mw_i, sat_model, host_model,\
                                                 disk_params, bulge_params,\
                                                 ac)

    return x_p, y_p, z_p, vx_p/conv_factor, vy_p/conv_factor, vz_p/conv_factor
