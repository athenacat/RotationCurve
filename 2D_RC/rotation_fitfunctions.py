
import numpy as np
import numpy.ma as ma

from scipy.optimize import minimize

#from RC_plotting_functions import plot_diagnostic_panel
from disk_mass import calc_mass_curve, fit_mass_curve
from rotation_curve_functions import bulge_vel, disk_vel, halo_vel_iso, halo_vel_NFW, halo_vel_Bur


H_0 = 100  # Hubble's Constant in units of h km/s/Mpc
c = 299792.458  # Speed of light in units of km/s



def find_phi(center_coords, phi_angle, vel_map):
    '''
    Find a point along the semi-major axis that has data to determine if phi
    needs to be adjusted.  (This is necessary because the positive y-axis is
    defined as being along the semi-major axis of the positive velocity side of
    the velocity map.)


    PARAMETERS
    ==========

    center_coords : tuple
        Coordinates of the center of the galaxy

    phi_angle : float
        Initial rotation angle of the galaxy, E of N.  Units are degrees.

    vel_map : masked ndarray of shape (n,n)
        Masked H-alpha velocity map


    RETURNS
    =======

    phi_adjusted : float
        Rotation angle of the galaxy, E of N, that points along the positive
        velocity sector.  Units are radians.
    '''

    # Convert phi_angle to radians
    phi = phi_angle * np.pi / 180.

    # phi = phi_angle

    # Extract "systemic" velocity (velocity at center spaxel)
    v_sys = vel_map[center_coords]

    print(v_sys)

    print(center_coords)

    print(phi)

    f = 0.4

    checkpoint_masked = True

    while checkpoint_masked:
        delta_x = int(center_coords[1] * f)
        delta_y = int(delta_x / np.tan(phi))
        semi_major_axis_spaxel = np.subtract(center_coords, (-delta_y, delta_x))

        '''
        print(center_coords)
        print(semi_major_axis_spaxel)
        '''

        for i in range(len(semi_major_axis_spaxel)):
            if semi_major_axis_spaxel[i] < 0:
                semi_major_axis_spaxel[i] = 0
            elif semi_major_axis_spaxel[i] >= vel_map.shape[i]:
                semi_major_axis_spaxel[i] = vel_map.shape[i] - 1
            # elif time.time() - start_time >= 1000:
            # lougoycheckpoint_masked = False

        # Check value along semi-major axis
        if vel_map.mask[tuple(semi_major_axis_spaxel)] == 0:
            checkpoint_masked = False
        # elif time.time() - start_time >= 3000:
        # checkpoint_masked = False
        else:
            f *= 0.9

    if vel_map[tuple(semi_major_axis_spaxel)] - v_sys < 0:
        phi_adjusted = phi + np.pi
    else:
        phi_adjusted = phi

    return phi_adjusted



def find_axis_ratio(incli):
    axis_ratio = np.sqrt((np.power(np.cos(incli),2) * (1-0.2**2)) + 0.2**2)
    return axis_ratio
    



def vel_tot_iso(r, params):
    rhob0, Rb, SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rh_pc = Rh * 1000

    Vbulge = bulge_vel(r, rhob0, Rb)
    Vdisk = disk_vel(r, SigD, Rd)
    Vhalo = halo_vel_iso(r_pc, rho0_h, Rh_pc)
    
    v2 = Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)


# -------------------------------------------------------------------------------
# NFW
# -------------------------------------------------------------------------------
def vel_tot_NFW(r, params):
    rhob0, Rb, SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rh_pc = Rh * 1000

    Vbulge = bulge_vel(r, rhob0, Rb)
    Vdisk = disk_vel(r, SigD, Rd)
    Vhalo = halo_vel_NFW(r_pc, rho0_h, Rh_pc)

    v2 = Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)


# -------------------------------------------------------------------------------
# Burket
# -------------------------------------------------------------------------------
def vel_tot_bur(r, params):
    rhob0, Rb, SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rh_pc = Rh * 1000

    Vbulge = bulge_vel(r, rhob0, Rb)
    Vdisk = disk_vel(r, SigD, Rd)
    Vhalo = halo_vel_Bur(r_pc, rho0_h, Rh_pc)
    v2 = Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)



def loglikelihood_iso_flat(params, rhob, Rb, SigD, Rd, scale, shape, vdata_flat, ivar_flat, mask):
    rho0_h, Rh, incl, phi, x_center, y_center, vsys = params
    total_param = [rhob, Rb, SigD, Rd, rho0_h, Rh, incl, phi, x_center, y_center, vsys]
    ############################################################################
    # Construct the model
    # ---------------------------------------------------------------------------
    model = rot_incl_iso(shape, scale, total_param)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    ############################################################################

    logL = -0.5 * np.sum((vdata_flat - model_flat) ** 2 * ivar_flat \
                         - np.log(ivar_flat))

    return logL


def nloglikelihood_iso_flat(params, rhob, Rb, SigD, Rd, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_iso_flat(params, rhob, Rb, SigD, Rd, scale, shape, vdata_flat, ivar_flat, mask)


def loglikelihood_NFW_flat(params, rhob, Rb, SigD, Rd, scale, shape, vdata_flat, ivar_flat, mask):
    rho0_h, Rh, incl, phi, x_center, y_center, vsys = params
    total_param = [rhob, Rb, SigD, Rd, rho0_h, Rh, incl, phi, x_center, y_center, vsys]
    ############################################################################
    # Construct the model
    # ---------------------------------------------------------------------------
    model = rot_incl_NFW(shape, scale, total_param)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    ############################################################################

    logL = -0.5 * np.sum((vdata_flat - model_flat) ** 2 * ivar_flat \
                         - np.log(ivar_flat))
    return logL


def nloglikelihood_NFW_flat(params, rhob, Rb, SigD, Rd, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_NFW_flat(params, rhob, Rb, SigD, Rd, scale, shape, vdata_flat, ivar_flat, mask)


def loglikelihood_bur_flat(params, rhob, Rb, SigD, Rd, scale, shape, vdata_flat, ivar_flat, mask):
    rho0_h, Rh, incl, phi, x_center, y_center, vsys = params
    total_param = [rhob, Rb, SigD, Rd, rho0_h, Rh, incl, phi, x_center, y_center, vsys]
    # log_rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys
    ############################################################################
    # Construct the model
    # ---------------------------------------------------------------------------
    model = rot_incl_bur(shape, scale, total_param)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    ############################################################################

    logL = -0.5 * np.sum((vdata_flat - model_flat) ** 2 * ivar_flat \
                         - np.log(ivar_flat))

    return logL


def nloglikelihood_bur_flat(params, rhob, Rb, SigD, Rd, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_bur_flat(params, rhob, Rb, SigD, Rd, scale, shape, vdata_flat, ivar_flat, mask)


def rot_incl_iso(shape, scale, params):
    rhob0, Rb, SigD, Rd, logrho0_h, Rh, inclination, phi, center_x, center_y, vsys = params
    rho0_h = np.power(10 ,logrho0_h)
    rotated_inclined_map = np.zeros(shape, dtype=float)

    for i in range(shape[0]):
        for j in range(shape[1]):
            x = ((j - center_x) * np.cos(phi) + np.sin(phi) * (i - center_y)) / np.cos(inclination)
            y = (-(j - center_x) * np.sin(phi) + np.cos(phi) * (i - center_y))

            r = np.sqrt(x ** 2 + y ** 2)

            theta = np.arctan2(-x, y)

            r_in_kpc = r * scale

            v_rot = vel_tot_iso(r_in_kpc, [rhob0, Rb, SigD, Rd, rho0_h, Rh])

            v = v_rot * np.sin(inclination) * np.cos(theta)

            rotated_inclined_map[i, j] = v + vsys

    return rotated_inclined_map


def rot_incl_NFW(shape, scale, params):
    rhob0, Rb, SigD, Rd, logrho0_h, Rh, inclination, phi, center_x, center_y, vsys = params
    rho0_h = np.power(10, logrho0_h)
    rotated_inclined_map = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            x = ((j - center_x) * np.cos(phi) + np.sin(phi) * (i - center_y)) / np.cos(inclination)
            y = (-(j - center_x) * np.sin(phi) + np.cos(phi) * (i - center_y))
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(-x, y)
            r_in_kpc = r * scale
            v = vel_tot_NFW(r_in_kpc, [rhob0, Rb, SigD, Rd, rho0_h, Rh]) * np.sin(inclination) * np.cos(theta)
            rotated_inclined_map[i, j] = v + vsys

    return rotated_inclined_map


def rot_incl_bur(shape, scale, params):
    rhob0, Rb, SigD, Rd, logrho0_h, Rh, inclination, phi, center_x, center_y, vsys = params
    rho0_h = np.power(10, logrho0_h)
    rotated_inclined_map = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            x = ((j - center_x) * np.cos(phi) + np.sin(phi) * (i - center_y)) / np.cos(inclination)
            y = (-(j - center_x) * np.sin(phi) + np.cos(phi) * (i - center_y))
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(-x, y)
            r_in_kpc = r * scale
            v = vel_tot_bur(r_in_kpc, [rhob0, Rb, SigD, Rd, rho0_h, Rh]) * np.sin(inclination) * np.cos(theta)
            rotated_inclined_map[i, j] = v + vsys
    
    return rotated_inclined_map


def parameterfit_iso(params, rhob, Rb, SigD, Rd, scale, shape, vmap, ivar, mask):
    
    rho_h, Rh, incl, ph, x_guess, y_guess, vsys = params
    print(incl)
    # Isothermal Fitting
    bounds_iso = [[-7, 2],  # Halo density [log(Msun/pc^3)]
                  [1, 1000],  # Halo radius [kpc]
                  [incl -(np.pi /6), 0.46 *np.pi],  # Inclination angle
                  [0, 2.2 * np.pi],  # Phase angle
                  [x_guess - 5, x_guess + 5],  # center_x
                  [y_guess - 5, y_guess + 5],  # center_y
                  [-100, 100]]  # systemic velocity

    vsys = 0
    print(bounds_iso)
    ig_iso = [rho_h, Rh, incl, ph, x_guess, y_guess, vsys]
    bestfit_iso = minimize(nloglikelihood_iso_flat,
                           ig_iso,
                           args=(rhob, Rb, SigD, Rd, scale, shape,
                                 vmap.compressed(),
                                 ivar.compressed(), mask),
                           method='Powell',
                           bounds=bounds_iso)
    print('---------------------------------------------------')
    print(bestfit_iso.status,bestfit_iso.x)

    return bestfit_iso.x


def parameterfit_NFW(params, rhob, Rb, SigD, Rd, scale, shape, vmap, ivar, mask):
    rho_h, Rh, incl, ph, x_guess, y_guess, vsys = params

    bounds_nfw = [[-4, 2],  # Halo density [log(Msun/pc^3)]
                  [0.1, 1000],  # Halo radius [kpc]
                  [0.1, 0.436 * np.pi],  # Inclination angle
                  [0, 2.2 * np.pi],  # Phase angle
                  [x_guess - 10, x_guess + 10],  # center_x
                  [y_guess - 10, y_guess + 10],  # center_y
                  [-100, 100]]  # systemic velocity

    vsys = 20

    ig_NFW = [rho_h, Rh, incl, ph, x_guess, y_guess, vsys]
    # ig_iso = [-1, 1, 1000, 4, 0.001, 25, incl, ph, x_guess, y_guess, vsys]
    # ig_iso = [0.0001, 4, 2000, 25, 5, 250, incl, ph, x_guess, y_guess, vsys]
    # print(ig_iso)

    bestfit_NFW = minimize(nloglikelihood_NFW_flat,
                           ig_NFW,
                           args=(rhob, Rb, SigD, Rd, scale, shape,
                                 vmap.compressed(),
                                 ivar.compressed(), mask),
                           method='Powell',
                           bounds=bounds_nfw)
    print('---------------------------------------------------')
    print(bestfit_NFW.status,bestfit_NFW.x)

    return bestfit_NFW.x


def parameterfit_bur(params, rhob, Rb, SigD, Rd, scale, shape, vmap, ivar, mask):
    rho_h, Rh, incl, ph, x_guess, y_guess, vsys = params

    bounds_bur = [[-7, 2],  # Halo density [log(Msun/pc^3)]
                  [0.1, 1000],  # Halo radius [kpc]
                  [incl -(np.pi /6), 0.46 *np.pi],  # Inclination angle
                  [0, 2.2 * np.pi],  # Phase angle
                  [x_guess - 10, x_guess + 10],  # center_x
                  [y_guess - 10, y_guess + 10],  # center_y
                  [-100, 100]]  # systemic velocity

    
    ig_bur = [rho_h, Rh, incl, ph, x_guess, y_guess, vsys]
    bestfit_bur = minimize(nloglikelihood_bur_flat,
                           ig_bur,
                           args=(rhob, Rb, SigD, Rd, scale, shape,
                                 vmap.compressed(),
                                 ivar.compressed(), mask),
                           method='Powell',
                           bounds=bounds_bur  )  # ,options = opts)
    #print('---------------------------------------------------')
    print(bestfit_bur.status,bestfit_bur.x)
    
    return bestfit_bur.x

def chi2(vmap, ivar, vmask, shape, scale, best_fit, fit_function):
    """
    Calculates the chi2 and reduced chi2 values of the total velocity curve fit
    
    Parameters:
    
    vmap: Masked Ha velocity map
    
    ivar: Masked array of the inverse variance of the Ha velocity map
    
    vmask: mask of the velocity map
    
    shape: shape of the velocity map
    
    scale: scale factor for converting spaxel radii to kpc
    
    best_fit: list of fitted parameters of velocity curve, 
        [rhob, Rb, SigD, Rd, rho0_h, Rh, incl, phi, x_center, y_center, vsys]
            rhob: central density of the bulge (M_sun/kpc^3)
            Rb: scale radius of the bulge (kpc)
            SigD: central density of the disk (M_sun/pc^2) 
            Rd:   scale radius of the disk (kpc)
            rho0_h: central density of the halo (log(M_sun/pc^3)
            Rh: scale radious of the halo (kpc)
            incl: inclination angle (radians)
            phi: rotation angle E of N (radians)
            x_center: x center spaxel
            y_center: y center spaxel
            vsys: systematic velocity of the galaxy (km/s)
        
    fit_function: halo model used in the fit. Options are "Isothermal", "NFW", and "Burkert"    
    
    
    """
    # Calculating the modeled velocity map
    
    if fit_function == "Isothermal":
        
        v_tot_map = ma.array(rot_incl_iso(shape, scale, best_fit), mask=vmask)
    
    elif fit_function == "NFW":
        
        v_tot_map = ma.array(rot_incl_NFW(shape, scale, best_fit), mask=vmask)
    
    elif fit_function == "Burkert":
        
        v_tot_map = ma.array(rot_incl_bur(shape, scale, best_fit), mask=vmask)
    else:
        print("Fit function not known")
    ###################################################
    
    x = ma.array((vmap - v_tot_map ) ** 2 * ivar ** 2, mask = vmask)
    n = x.count()
    chi2 = ma.sum(x)
    chi2r = chi2 / (n - 7)
    
    return chi2, chi2r


    
    

        
    