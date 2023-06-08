# from Kelly Douglass

import numpy as np
import numpy.ma as ma
from galaxy_component_functions import bulge_vel, \
    disk_vel, \
    halo_vel_iso, \
    halo_vel_NFW, \
    halo_vel_bur, \
    vel_tot_iso, \
    vel_tot_NFW, \
    vel_tot_bur
from scipy.optimize import minimize, Bounds

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


def loglikelihood_iso_flat(params, rhob, Rb, SigD, Rd, scale, shape, vdata_flat, ivar_flat, mask):
    rho0_h, Rh, incl, phi, x_center, y_center, vsys = params
    total_param = [rhob, Rb, SigD, Rd,rho0_h, Rh,incl,phi,x_center,y_center, vsys]
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
    total_param = [rhob, Rb, SigD, Rd] + params
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
    total_param = [rhob, Rb, SigD, Rd] + params
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
    rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys = params

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
    rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys = params
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
    rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys = params
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
    incl, ph, x_guess, y_guess = params
    # Isothermal Fitting
    bounds_iso = [[-7, 2],  # Halo density [log(Msun/pc^3)]
                  [0.1, 1000],  # Halo radius [kpc]
                  [0.1, 0.436 * np.pi],  # Inclination angle
                  [0, 2.2 * np.pi],  # Phase angle
                  [x_guess - 10, x_guess + 10],  # center_x
                  [y_guess - 10, y_guess + 10],  # center_y
                  [-100, 100]]  # systemic velocity

    vsys = 0

    ig_iso = [-3, 25, incl, ph, x_guess, y_guess, vsys]
    # ig_iso = [-1, 1, 1000, 4, 0.001, 25, incl, ph, x_guess, y_guess, vsys]
    # ig_iso = [0.0001, 4, 2000, 25, 5, 250, incl, ph, x_guess, y_guess, vsys]
    print(ig_iso)

    bestfit_iso = minimize(nloglikelihood_iso_flat,
                           ig_iso,
                           args=(rhob, Rb, SigD, Rd, scale, shape,
                                 vmap.compressed(),
                                 ivar.compressed(), mask),
                           method='Powell',
                           bounds=bounds_iso)
    print('---------------------------------------------------')
    print(bestfit_iso)

    return bestfit_iso.x
