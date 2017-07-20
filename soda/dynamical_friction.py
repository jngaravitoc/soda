import numpy as np
from scipy.special import erf
from astropy import units, constants
from .profiles import *
#from adiabatic_contraction import rho_ac
from scipy.integrate import quad

# Coulomb Logarithm definition:
# See:
def coulomb_log(r, alpha):
    bmax = r # position of test particle at a given time
    # k = softening length if the satellite is modeled with a plummer
    # profile. See http://adsabs.harvard.edu/abs/2007ApJ...668..949B
    k = 3.0 #kpc
    #k = rs_sat # be careful with this! <-----------------
    bmin = 1.4 * k
    L = bmax / bmin

    # alpha is for make the dynamical friction more realistic.
    CL = alpha * np.log(L)
    return CL

def coulomb_tremaine_cdm(r, v, M_sat, alpha):
    # See https://arxiv.org/abs/1610.08297
    # Ultraligh scalar DM dynamical friction fudge term quantum mechanics -> classical approx.

    r = r*units.kpc
    G1 = G.to(units.kpc**3 / units.Msun / units.Gyr**2)
    #print(2*v**2*r/(G1*M_sat))
    CL = alpha * np.log((2*v**2*r/(G1*M_sat)).value)
    return np.float32(CL)

def integrand(t):
    # integral -> This can be done analytically! 
    y= (1-np.cos(t))/t
    return y

def coulomb_tremaine_uldm(r, v, M_sat, alpha, M_scalar):
    # Ultraligh scalar DM dynamical friction fudge term quantum mechanics.
    
    # See https://arxiv.org/abs/1610.08297
    #M_scalar = 1E-22 * units.eV
    
    M_scalar = M_scalar * units.eV
    hbarc2 = 1.9199E-18 * units.eV * units.km * units.kpc / units.s
    k1 = M_scalar * v / (hbarc2)
    k1 = k1.to(1/units.kpc)

    cin = quad(integrand, 0, (2*k1*r).value)
    CL = cin[0] + (np.sin((2*k1*r).value)/(2*k1*r).value) - 1
    return alpha*CL

# Coulomb Logarithm from Van Der Marel et al 2013. Eq A
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
       M_bulge, ac, alpha, C):
    """
    Function that computes the dynamical friction
    of a satellite around the host halo.

    parameters:

    x, y, z:

    vx, vy, vz

    M1, M2

    Rv

    c

    host_model

    M_disk

    M_bulge

    ac

    alpha

    C : Lambda = bmax/bmin (0), Tremaine CDM (1), Tremaine Ultralight Scalar (2).

    """

    # M2 would be the galaxy feeling the dynamical friction due to M1
    # Rv, c, (x, y, z) and (vx, vy, vz) are for the M2 galaxy
    # Coordinates
    r = np.sqrt(x**2.0 + y**2.0 + z**2.0)

    # Velocities
    v = np.sqrt(vx**2.0 + vy**2.0 + vz**2.0)
    v = v * units.kpc / units.Gyr

    # Computing the density of the host galaxy at r
    
    ## With adiabatic contraction.
    if (ac==1):
        rho = rho_ac(r) # using the adiabatic contraction interpolated density
        #rho = dens_NFWnRvir(c, x, y, z, M1, Rv)
    # NFW 
    elif ((host_model[0] == 'NFW') & (ac==0)):
        rho = dens_NFWnRvir(c, x, y, z, M1, Rv)
    # triaxial NFW
    elif ((host_model[0] == 'NFW_T') & (ac==0)):
        q = host_model[4]
        s = host_model[5]
        rho = dens_NFWnRvir_T(c, x, y, z, M1, Rv, q, s)
    # Hernquist
    elif ((host_model[0] == 'hernquist') & (ac==0)):
        rho = dens_hernquist(Rv, x, y, z, M1) # Rv is a in this case
    # plummer
    elif ((host_model[0] == 'plummer') & (ac==0)):
        rho = dens_plummer(Rv, x, y, z, M1) # Rv is a in this case

    rho = rho * units.Msun / units.kpc**3.0

    # Constants
    forpiG2 = - 4.0 * np.pi * G**2.0
    forpiG2 = forpiG2.to(units.kpc**6.0 / units.Msun**2.0 / units.Gyr**4.0)

    #factor = factor.value
    vx = vx * units.kpc / units.Gyr
    vy = vy * units.kpc / units.Gyr
    vz = vz * units.kpc / units.Gyr
    M2 = M2 * units.Msun
    M1 = M1 * units.Msun
    
    # fudge factor in dynamical friction
    if (alpha[0]==0):

         if C==0: # Regular dm LAMBDA = bmax/bmin
             Coulomb = coulomb_log(r, alpha[1])

         elif C==1: # Tremaine regular DM.
             Coulomb = coulomb_tremaine_cdm(r, v, M2, alpha[1])
             print(alpha[1], Coulomb)

         elif C==2: # Tremaine ultraligh scalar.
             Coulomb = coulomb_tremaine_uldm(r, v, M2, alpha[1], alpha[2])

    # Van Der Marel dynamical friction fudge factor.
    elif (alpha[0]==1):
         L = alpha[2]
         C = alpha[3]
         Coulomb = coulomb_v_log(L, r, alpha[1], rs_sat ,C)

    # sigma term
            
    if host_model[0]=='hernquist':
        s = sigma(c, r, M1.value - M_disk - M_bulge, Rv*c)
    elif host_model[0]=='plummer':
        s = sigma(c, r, M1.value - M_disk - M_bulge, Rv*c)
    else:
        s = sigma(c, r, M1.value - M_disk - M_bulge, Rv)
    X = v/(np.sqrt(2.0)*s)

    # Dynamical friction acceleration terms:
    # Usual with error function.
    if C==0:
        a_dfx = (forpiG2*M2*rho*Coulomb*(erf(X.value) - 2.0*X/(np.sqrt(np.pi))\
                 *np.exp(-X**2.0))*vx)/v**3.0
        a_dfy = (forpiG2*M2*rho*Coulomb*(erf(X.value) - 2.0*X/(np.sqrt(np.pi))\
                 *np.exp(-X**2.0))*vy)/v**3.0
        a_dfz = (forpiG2*M2*rho*Coulomb*(erf(X.value) - 2.0*X/(np.sqrt(np.pi))\
                 *np.exp(-X**2.0))*vz)/v**3.0

    # Quantum formulation:
    if ((C==1) | (C==2)):
        a_dfx = (forpiG2*M2*rho*Coulomb)*vx/v**3.0
        a_dfy = (forpiG2*M2*rho*Coulomb)*vy/v**3.0
        a_dfz = (forpiG2*M2*rho*Coulomb)*vz/v**3.0

    # Units conversion
    a_dfx = a_dfx.to(units.kpc / units.Gyr**2.0)
    a_dfy = a_dfy.to(units.kpc / units.Gyr**2.0)
    a_dfz = a_dfz.to(units.kpc / units.Gyr**2.0)

    return a_dfx.value, a_dfy.value, a_dfz.value
