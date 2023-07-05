import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

from astropy.io import fits
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma


from astropy.io import fits
from disk_mass_plotting_functions import plot_fitted_disk_rot_curve
from disk_mass_functions import chi2_mass

import sys
sys.path.insert(1,"main/")
from galaxy_component_functions_cython import vel_tot_bur, vel_tot_iso, vel_tot_NFW, disk_vel, bulge_vel, halo_vel_NFW, halo_vel_bur, halo_vel_iso
from Velocity_Map_Functions_cython import rot_incl_iso, rot_incl_bur, rot_incl_NFW


def Plotting_Isothermal(ID, shape, scale, fit_solution, mask, ax=None):
    '''

    :param ID:
    :param shape:
    :param scale:
    :param fit_solution:
    :param mask:
    :return:
    '''

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    print(fit_solution)
    iso_map = ax.imshow(ma.array(rot_incl_iso(shape, scale, fit_solution), mask=mask),
                        origin='lower',
                        cmap='RdBu_r')

    ax.set_title(ID + ' Isothermal Fit')

    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')

    cbar = plt.colorbar(iso_map, ax=ax)
    cbar.set_label('km/s')

    # plt.close()


# NFW
def Plotting_NFW(ID, shape, scale, fit_solution, mask, ax=None):
    '''

    :param ID:
    :param shape:
    :param scale:
    :param fit_solution:
    :param mask:
    :return:
    '''

    if ax is None:
        fig, ax = plt.subplots()

    NFW_map = ax.imshow(ma.array(rot_incl_NFW(shape, scale, fit_solution), mask=mask),
                        origin='lower',
                        cmap='RdBu_r')

    ax.set_title(ID + ' NFW Fit')

    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')

    cbar = plt.colorbar(NFW_map, ax=ax)
    cbar.set_label('km/s')

    # plt.close()


# Burket
def Plotting_Burkert(ID, shape, scale, fit_solution, mask, ax=None):
    '''

    :param ID:
    :param shape:
    :param scale:
    :param fit_solution:
    :param mask:
    :return:
    '''

    if ax is None:
        fig, ax = plt.subplots()

    bur_map = ax.imshow(ma.array(rot_incl_bur(shape, scale, fit_solution), mask=mask),
                        origin='lower',
                        cmap='RdBu_r')

    ax.set_title(ID + ' Burkert Fit')

    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')

    cbar = plt.colorbar(bur_map, ax=ax)
    cbar.set_label('km/s')

    # plt.savefig(ID + ' Burket.png', format='png')
    # plt.close()


################################################################################
def deproject_spaxel(coords, center, phi, i_angle):
    '''
    Calculate the deprojected radius for the given coordinates in the map.


    PARAMETERS
    ==========

    coords : length-2 tuple
        (i,j) coordinates of the current spaxel

    center : length-2 tuple
        (i,j) coordinates of the galaxy's center

    phi : float
        Rotation angle (in radians) east of north of the semi-major axis.

    i_angle : float
        Inclination angle (in radians) of the galaxy.


    RETURNS
    =======

    r : float
        De-projected radius from the center of the galaxy for the given spaxel
        coordinates.
    '''

    # Distance components between center and current location
    delta = np.subtract(coords, center)

    # x-direction distance relative to the semi-major axis
    dx_prime = (delta[1] * np.cos(phi) + delta[0] * np.sin(phi)) / np.cos(i_angle)

    # y-direction distance relative to the semi-major axis
    dy_prime = (-delta[1] * np.sin(phi) + delta[0] * np.cos(phi))

    # De-projected radius for the current point
    r = np.sqrt(dx_prime ** 2 + dy_prime ** 2)

    # Angle (counterclockwise) between North and current position
    theta = np.arctan2(-dx_prime, dy_prime)

    return r, theta


################################################################################


################################################################################
def plot_rot_curve(mHa_vel,
                   mHa_vel_ivar,
                   best_fit_values,
                   scale,
                   gal_ID,
                   halo_model,
                   IMAGE_DIR=None,
                   IMAGE_FORMAT='jpg',
                   ax=None):
    '''
    Plot the galaxy rotation curve.


    PARAMETERS
    ==========

    mHa_vel : numpy ndarray of shape (n,n)
        Masked H-alpha velocity array

    mHa_vel_ivar : numpy ndarray of shape (n,n)
        Masked array of the inverse variance of the H-alpha velocity
        measurements

    best_fit_values : dictionary
        Best-fit values for the velocity map

    scale : float
        Pixel scale (to convert from pixels to kpc)

    gal_ID : string
        MaNGA <plate>-<IFU> for the current galaxy

    halo_model : string
        Determines which function to use for the velocity.  Options are 'Isothermal','NFW', and
        'Burkert'.

    IMAGE_DIR : str
        Path of directory in which to store plot.
        Default is None (image will not be saved)

    IMAGE_FORMAT : str
        Format of saved plot
        Default is 'eps'

    ax : matplotlib.pyplot figure axis object
        Axis handle on which to create plot
    '''
    print(best_fit_values)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    ############################################################################
    # Extract inclination angle (in radians)
    # ---------------------------------------------------------------------------
    i_angle = best_fit_values[6]  # np.arccos(best_fit_values['ba'])
    ############################################################################

    ############################################################################
    # Extract rotation angle (in radians)
    # ---------------------------------------------------------------------------
    phi = best_fit_values[7]
    ############################################################################
    #[rhob, Rb, SigD, Rd, rho0_h, Rh, incl, phi, x_center, y_center, vsys]
    ############################################################################
    # Deproject all data values in the given velocity map
    # ---------------------------------------------------------------------------
    vel_array_shape = mHa_vel.shape

    r_deproj = np.zeros(vel_array_shape)
    v_deproj = np.zeros(vel_array_shape)

    theta = np.zeros(vel_array_shape)
    
    for i in range(vel_array_shape[0]):
        for j in range(vel_array_shape[1]):

            r_deproj[i, j], theta[i, j] = deproject_spaxel((i, j),
                                                           (best_fit_values[8], best_fit_values[9]),
                                                           phi,
                                                           i_angle)

            ####################################################################
            # Find the sign of r_deproj
            # -------------------------------------------------------------------
            if np.cos(theta[i, j]) < 0:
                r_deproj[i, j] *= -1
            ####################################################################

    # Scale radii to convert from spaxels to kpc
    r_deproj *= scale

    # Deproject velocity values
    
    v_deproj = (mHa_vel - best_fit_values[10]) / np.abs(np.cos(theta))
    v_deproj /= np.sin(i_angle)

    # Apply mask to arrays
    rm_deproj = ma.array(r_deproj, mask=mHa_vel.mask)
    vm_deproj = ma.array(v_deproj, mask=mHa_vel.mask)
    ############################################################################

    ############################################################################
    # Calculate functional form of rotation curve
    # ---------------------------------------------------------------------------
    #r = np.linspace(ma.min(rm_deproj), ma.max(rm_deproj), 100)
    r = np.linspace(0, ma.max(rm_deproj), 100)
    
    
    
    v_b = np.zeros(len(r))
    v_d = np.zeros(len(r))
    v_h = np.zeros(len(r))
    v = np.zeros(len(r))
    
    
    for i in range(len(r)):
    
        #v_b = np.zeros(len(r))
        v_b[i] = bulge_vel(r[i]*1000, best_fit_values[0], best_fit_values[1]*1000)
        
        #v_d = np.zeros(len(r))
        v_d[i] = disk_vel(r[i]*1000, best_fit_values[2], best_fit_values[3]*1000)
        
        if halo_model == 'Isothermal':
            v_h[i] = halo_vel_iso(r[i]*1000,best_fit_values[4], best_fit_values[5] * 1000)
        elif halo_model == 'NFW':
            v_h[i] = halo_vel_NFW(r[i]*1000,best_fit_values[4], best_fit_values[5] * 1000)
        elif halo_model == 'Burkert':
            v_h[i] = halo_vel_bur(r[i]*1000,best_fit_values[4], best_fit_values[5] * 1000)
        else:
            print('Fit function not known.  Please update plot_rot_curve function.')
        
        v[i] = np.sqrt(v_b[i]**2 + v_d[i]**2 + v_h[i]**2)
    #v_h = np.zeros(len(r))
    
    #v = np.zeros(len(r))
    
    ############################################################################

    ############################################################################
    # Plot rotation curve
    # ---------------------------------------------------------------------------
    ax.set_title(gal_ID + ' ' + halo_model)

    ax.plot(rm_deproj, vm_deproj, 'k.', markersize=1)
    ax.plot(np.concatenate((-np.flip(r), r)), np.concatenate((-np.flip(v), v)), 'c', label='$v_{tot}$')
    ax.plot(np.concatenate((-np.flip(r), r)), np.concatenate((-np.flip(v_b), v_b)), '--', label='bulge')
    
    ax.plot(np.concatenate((-np.flip(r), r)), np.concatenate((-np.flip(v_d), v_d)), '-.', label='disk')
    ax.plot(np.concatenate((-np.flip(r), r)), np.concatenate((-np.flip(v_h), v_h)), ':', label='halo')
    
    
    vmax = np.nanmax(v)
    print(vmax)
    if np.isfinite(vmax):
        ax.set_ylim([-1.25 * vmax, 1.25 * vmax])
        ax.tick_params(axis='both', direction='in')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_xlabel('Deprojected radius [kpc/h]')
        ax.set_ylabel('Rotational velocity [km/s]')
        plt.legend()
    else:
        vmax = 1000
        ax.set_ylim([-vmax, vmax])
        ax.tick_params(axis='both', direction='in')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_xlabel('Deprojected radius [kpc/h]')
        ax.set_ylabel('Rotational velocity [km/s]')
        plt.legend()
    # plt.savefig(gal_ID + ' rotation curve ' + halo_model + '.png',format='png')
    ############################################################################





def plot_diagnostic_panel(ID, shape, scale, Isothermal_Fit, NFW_Fit, Burket_Fit, mask, vmasked, ivar_masked ):
    '''
    Plot a set of 6 plots: modeled 2d velocity maps and decomposed velocity vs deprojected radius plots for all three halo models


    Parameters
    ==========

    gal_ID : string
        MaNGA plate number - MaNGA fiberID number

    shape: float
        shape of the velocity map
        
    scale: float
        conversion factor from spaxel to kpc
        
    Isothermal_Fit: dictionary
        isothermal best fit values
    
    NFW_Fit : dictionary
        NFW best fit values
    
    Burket_Fit: dictionary
        Burkert best fit values

    vmasked : numpy array
        Masked H-alpha velocity map

    ivar_masked: numpy array
        Masked inverse variance of H-alpha velocity map
    
    
    
    '''

    #    panel_fig, (( Ha_vel_panel, mHa_vel_panel),
    #                ( contour_panel, rot_curve_panel)) = plt.subplots( 2, 2)
    panel_fig, ((Isothermal_Plot_panel, NFW_Plot_panel, Burket_Plot_panel),
                (RC_Isothermal, RC_NFW, RC_Burket)) = plt.subplots(2, 3)
    panel_fig.set_figheight(10)
    panel_fig.set_figwidth(15)
    plt.suptitle(ID + " Diagnostic Panel", y=1.05, fontsize=16)

    Plotting_Isothermal(ID, shape, scale, Isothermal_Fit, mask, ax=Isothermal_Plot_panel)

    Plotting_NFW(ID, shape, scale, NFW_Fit, mask, ax=NFW_Plot_panel)

    Plotting_Burkert(ID, shape, scale, Burket_Fit, mask, ax=Burket_Plot_panel)
    print("Iso:",Isothermal_Fit)
    plot_rot_curve(vmasked, ivar_masked, Isothermal_Fit, scale, ID, 'Isothermal', ax=RC_Isothermal)
    print("NFW",NFW_Fit)
    plot_rot_curve(vmasked, ivar_masked, NFW_Fit, scale, ID, 'NFW', ax=RC_NFW)
    print("Bur:",Burket_Fit)
    plot_rot_curve(vmasked, ivar_masked, Burket_Fit, scale, ID, 'Burkert', ax=RC_Burket)

    panel_fig.tight_layout()

    plt.savefig(ID + '_Diagnostic_Panels_new')

    
def plot_totfit_panel (ID, shape, scale, vfit, fit_function, vmask, vmasked, ivar_masked, mass_data, mass_params):
    #Set of four plots: actual velocity map, modeled 2D velocity map, disk mass fit, and decomposed v vs deprojected radius for a given halo model
    
    ###find chi2 of the disk mass fit
    sM_chi2 = chi2_mass([mass_params['rho_bulge'], mass_params['R_bulge'], \
                         mass_params['Sigma_disk'], mass_params['R_disk']], \
                         mass_data['radius'], \
                         mass_data['star_vel'], mass_data['star_vel_err'])
    print(sM_chi2)
    panel_fig, ((vmap_panel,sMass_panel),(fit_panel, deproj_panel)) = plt.subplots(2,2)
    panel_fig.set_figheight(10)
    panel_fig.set_figwidth(15)
    plt.suptitle(ID + " Diagnostic Panel", y=1.05, fontsize=16)
    
    vmap = vmap_panel.imshow(vmasked, origin='lower', cmap='RdBu_r')
    vmap_panel.set_title(ID + ' Ha Map')
    vmap_panel.set_xlabel('spaxel')
    vmap_panel.set_ylabel('spaxel')
    cbar = plt.colorbar(vmap, ax=vmap_panel)
    cbar.set_label('km/s')
    
    plot_fitted_disk_rot_curve(ID, mass_data,mass_params,sM_chi2,'bulge',ax=sMass_panel)
    
    if fit_function == "Isothermal":
        Plotting_Isothermal(ID,shape,scale,vfit,vmask,ax = fit_panel)
        plot_rot_curve(vmasked,ivar_masked,vfit,scale,ID,"Isothermal",ax=deproj_panel)
    elif fit_function == "NFW":
        Plotting_NFW(ID,shape,scale,vfit,vmask,ax = fit_panel)
        plot_rot_curve(vmasked,ivar_masked,vfit,scale,ID,"NFW",ax=deproj_panel)
    elif fit_function == "Burkert":
        Plotting_Burkert(ID,shape,scale,vfit,vmask,ax = fit_panel)
        plot_rot_curve(vmasked,ivar_masked,vfit,scale,ID,"Burkert",ax=deproj_panel)
    else:
        print("Fit function not recognized")
    
    panel_fig.tight_layout()
    
    plt.savefig(ID + "_" + fit_function + "_diagonistic")