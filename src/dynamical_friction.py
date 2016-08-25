import numpy as np
from scipy.special import erf
from astropy import units, constants
from profiles import *
#from adiabatic_contraction import rho_ac

# Coulomb Logarithm definition:
def coulomb_log(r, alpha):
    bmax = r # position of test particle at a given time
    # k = softening length if the satellite is modeled with a plummer
    # profile. See http://adsabs.harvard.edu/abs/2007ApJ...668..949B
    k = 3.0 #kpc
    #k = rs_sat
    bmin = 1.4 * k
    L = bmax / bmin
    # alpha is for make the dynamical friction more realistic.
    CL = alpha * np.log(L)
    return CL

# Coulomb Logarithm from Van Der Marel et al 2013. Eq A1
def coulomb_v_log(L, r, alpha_v, a, C):
    l = np.log(r/(C*a))**alpha_v
    x = [l, L]
    return np.max(x)


#One dimensional velocidty dispersion analytic approx.
#From Zentner and Bullock 2003 for a NFW profile!
def sigma(c, r, M, Rv):
    M = M * units.Msun
    Rv = Rv * units.kpc
    vvir = np.sqrt(G * M / Rv)
    g = np.log(1+c) - (c /(1+c))
    vmax = np.sqrt(0.216 * vvir**2 * c / g)
    x = r / Rv.value * c
    sigma = vmax * 1.4393 * x **(0.354) / (1.0 + 1.1756*x**0.725)
    sigma = sigma.to(units.kpc / units.Gyr)
    return sigma

# Dynamical Friction computation
def df(x, y, z, vx, vy, vz, M1, M2, Rv, c, host_model, M_disk, \
       M_bulge, ac, alpha):
    """
    Function that computes the dynamical friction
    of a satellite around the host halo.

    
    """



    # M2 would be the galaxy feeling the dynamical friction due to M1
    # Rv, c, (x, y, z) and (vx, vy, vz) are for the M2 galaxy
    # Coordinates
    r = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    # Velocities
    v = np.sqrt(vx**2.0 + vy**2.0 + vz**2.0)
    v = v * units.kpc / units.Gyr
    # Density of the NFW at a given r
    if (ac==1):
        rho = rho_ac(r) # using the adiabatic contraction interpolated density
        #rho = dens_NFWnRvir(c, x, y, z, M1, Rv)
    elif ((host_model[0] == 'NFW') & (ac==0)) :
        rho = dens_NFWnRvir(c, x, y, z, M1, Rv)
    elif ((host_model[0] == 'hernquist') & (ac==0)):
        rho = dens_hernquist(rs_host, x, y, z, M1)
    rho = rho * units.Msun / units.kpc**3.0
    # Computing the dynamical friction
    factor = - 4.0 * np.pi * G**2.0
    factor = factor.to(units.kpc**6.0 / units.Msun**2.0 / units.Gyr**4.0)
    #factor = factor.value
    vx = vx * units.kpc / units.Gyr
    vy = vy * units.kpc / units.Gyr
    vz = vz * units.kpc / units.Gyr
    M2 = M2 * units.Msun
    M1 = M1 * units.Msun

    if (alpha[0]==0):
         Coulomb = coulomb_log(r, alpha[1])
    elif (alpha[1]==1):
         L = alpha[2]
         C = alpha[3]
         Coulomb = coulomb_v_log(L, r, alpha[1], rs_sat ,C)
    s = sigma(c, r, M1.value - M_disk - M_bulge, Rv) # check this !!!
    #s = sigma(c, r, M1.value, Rv)
    X = v / ( np.sqrt(2.0) * s)
    # Main equation
    a_dfx = (factor*M2*rho*Coulomb*(erf(X.value) - 2.0*X/(np.sqrt(np.pi))*np.exp(-X**2.0))*vx)/v**3.0
    a_dfy = (factor*M2*rho*Coulomb*(erf(X.value) - 2.0*X/(np.sqrt(np.pi))*np.exp(-X**2.0))*vy)/v**3.0
    a_dfz = (factor*M2*rho*Coulomb*(erf(X.value) - 2.0*X/(np.sqrt(np.pi))*np.exp(-X**2.0))*vz)/v**3.0
    # Conversion to the correct units
    a_dfx = a_dfx.to(units.kpc / units.Gyr**2.0)
    a_dfy = a_dfy.to(units.kpc / units.Gyr**2.0)
    a_dfz = a_dfz.to(units.kpc / units.Gyr**2.0)
    return a_dfx.value, a_dfy.value, a_dfz.value
