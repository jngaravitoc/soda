import numpy as np
from astropy import units, constants
from acceleration import *

def integrate(time, pos_sat, vel_sat, pos_host, vel_host, host_model, \
             sat_model, disk_params, bulge_params, ac=0, \
             dfric=1, alpha=0, host_move=1, direction=1, dt=0.01):

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

    TO-DO:
    ------

    1. Generalize for N satellites.
    2. Integrate with galpy/gala
    3. Used in arbitrary accelerations/SCF
    """

    conv_factor = 1.0227121650537077
    # h is the time step
    h = dt * direction
    n_points = int(time / dt) # Make this an input parameter!

    t = np.zeros(n_points)
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)

    x_mw = np.zeros(n_points)
    y_mw = np.zeros(n_points)
    z_mw = np.zeros(n_points)

    vx = np.zeros(n_points)
    vy = np.zeros(n_points)
    vz = np.zeros(n_points)

    vx_mw  = np.zeros(n_points)
    vy_mw  = np.zeros(n_points)
    vz_mw  = np.zeros(n_points)

    ax = np.zeros(n_points)
    ay = np.zeros(n_points)
    az = np.zeros(n_points)

    ax_mw  = np.zeros(n_points)
    ay_mw  = np.zeros(n_points)
    az_mw  = np.zeros(n_points)

    t[0] = 0 # Make this an input parameter?
    x[0] = pos_sat[0]
    y[0] = pos_sat[1]
    z[0] = pos_sat[2]

    x_mw[0] = pos_host[0]
    y_mw[0] = pos_host[1]
    z_mw[0] = pos_host[2]

    vx[0] = vel_sat[0]*conv_factor
    vy[0] = vel_sat[1]*conv_factor
    vz[0] = vel_sat[2]*conv_factor

    vx_mw[0] = vel_host[0]*conv_factor
    vy_mw[0] = vel_host[1]*conv_factor
    vz_mw[0] = vel_host[2]*conv_factor

    pos_0 = np.array([x[0]-x_mw[0], y[0]-y_mw[0], z[0]-z_mw[0]])
    vel_0 = np.array([vx[0]-vx_mw[0], vy[0]-vy_mw[0], vz[0]-vz_mw[0]])

    ax[0] = acc_sat(pos_0, vel_0, host_model, sat_model, \
                    disk_params, bulge_params, ac, dfric, alpha)[0]

    ay[0] = acc_sat(pos_0, vel_0, host_model, sat_model, \
                    disk_params, bulge_params, ac, dfric, alpha)[1]

    az[0] = acc_sat(pos_0, vel_0, host_model, sat_model, \
                    disk_params, bulge_params, ac, dfric, alpha)[2]

    ax_mw[0] = acc_host(-pos_0, -vel_0, host_model, sat_model)[0]
    ay_mw[0] = acc_host(-pos_0, -vel_0, host_model, sat_model)[1]
    az_mw[0] = acc_host(-pos_0, -vel_0, host_model, sat_model)[2]

    # half step
    # Here I assume the host galaxy starts at position (0, 0, 0) and then its
    #print(ax[0], ay[0], az[0], ax_mw[0], ay_mw[0], az_mw[0])
    #print(vx[0], vy[0], vz[0], vx_mw[0], vy_mw[0], vz_mw[0])

    # initial v[1] is (0, 0, 0)
    t[1] = t[0] - h
    x[1] = x[0] - h * vx[0]
    y[1] = y[0] - h * vy[0]
    z[1] = z[0] - h * vz[0]

    vx[1] = vx[0] - h * ax[0]
    vy[1] = vy[0] - h * ay[0]
    vz[1] = vz[0] - h * az[0]

    pos_1 = np.array([x[1]-x_mw[1], y[1]-y_mw[1], z[1]-z_mw[1]])
    vel_1 = np.array([vx[1]-vx_mw[1], vy[1]-vy_mw[1], vz[1]-vz_mw[1]])

    if (host_move==1):#--------------------------------
        x_mw[1] = x_mw[0] - h * vx_mw[0]
        y_mw[1] = y_mw[0] - h * vy_mw[0]
        z_mw[1] = z_mw[0] - h * vz_mw[0]

        vx_mw[1] = vx_mw[0] - h * ax_mw[0]
        vy_mw[1] = vy_mw[0] - h * ay_mw[0]
        vz_mw[1] = vz_mw[0] - h * az_mw[0]

        pos_1 = np.array([x[1]-x_mw[1], y[1]-y_mw[1], z[1]-z_mw[1]])
        vel_1 = np.array([vx[1]-vx_mw[1], vy[1]-vy_mw[1], vz[1]-vz_mw[1]])

        ax_mw[1] = acc_host(-pos_1, -vel_1, host_model, sat_model)[0]
        ay_mw[1] = acc_host(-pos_1, -vel_1, host_model, sat_model)[1]
        az_mw[1] = acc_host(-pos_1, -vel_1, host_model, sat_model)[2]

    ax[1] = acc_sat(pos_1, vel_1, host_model, sat_model\
                   ,disk_params, bulge_params, ac, dfric, alpha)[0]
    ay[1] = acc_sat(pos_1, vel_1, host_model, sat_model\
                   ,disk_params, bulge_params, ac, dfric, alpha)[1]
    az[1] = acc_sat(pos_1, vel_1, host_model, sat_model\
                   ,disk_params, bulge_params, ac, dfric, alpha)[2]

    #print(ax[1], ay[1], az[1], ax_mw[1], ay_mw[1], az_mw[1])
    #print(vx[1], vy[1], vz[1], vx_mw[1], vy_mw[1], vz_mw[1])

    for i in range(2, len(x)):
        t[i] = t[i-1] - h
        x[i] = x[i-2] - 2 * h * vx[i-1]
        y[i] = y[i-2] - 2 * h * vy[i-1]
        z[i] = z[i-2] - 2 * h * vz[i-1]

        vx[i] = vx[i-2] - 2 * h * ax[i-1]
        vy[i] = vy[i-2] - 2 * h * ay[i-1]
        vz[i] = vz[i-2] - 2 * h * az[i-1]

        pos_i = np.array([x[i]-x_mw[i], y[i]-y_mw[i], z[i]-z_mw[i]])
        vel_i = np.array([vx[i]-vx_mw[i], vy[i]-vy_mw[i], vz[i]-vz_mw[i]])

        if (host_move==1):
            x_mw[i] = x_mw[i-2] - 2 * h * vx_mw[i-1]
            y_mw[i] = y_mw[i-2] - 2 * h * vy_mw[i-1]
            z_mw[i] = z_mw[i-2] - 2 * h * vz_mw[i-1]

            vx_mw[i] = vx_mw[i-2] - 2 * h * ax_mw[i-1]
            vy_mw[i] = vy_mw[i-2] - 2 * h * ay_mw[i-1]
            vz_mw[i] = vz_mw[i-2] - 2 * h * az_mw[i-1]

            pos_i = np.array([x[i]-x_mw[i], y[i]-y_mw[i], z[i]-z_mw[i]])
            vel_i = np.array([vx[i]-vx_mw[i], vy[i]-vy_mw[i], vz[i]-vz_mw[i]])

            ax_mw[i] = acc_host(-pos_i, -vel_i, host_model, sat_model)[0]
            ay_mw[i] = acc_host(-pos_i, -vel_i, host_model, sat_model)[1]
            az_mw[i] = acc_host(-pos_i, -vel_i, host_model, sat_model)[2]

        ax[i] = acc_sat(pos_i, vel_i, host_model, sat_model\
                       ,disk_params, bulge_params, ac, dfric, alpha)[0]
        ay[i] = acc_sat(pos_i, vel_i, host_model, sat_model\
                       ,disk_params, bulge_params, ac, dfric, alpha)[1]
        az[i] = acc_sat(pos_i, vel_i, host_model, sat_model\
                       ,disk_params, bulge_params, ac, dfric, alpha)[2]

    return t, np.array([x, y, z]).T, np.array([vx, vy, vz]).T/conv_factor, \
           np.array([x_mw, y_mw, z_mw]).T, np.array([vx_mw, vy_mw,vz_mw]).T/conv_factor
