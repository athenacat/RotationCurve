import numpy as np
import numpy.ma as ma
import RC_plotting_functions as RC
from disk_mass import calc_mass_curve, fit_mass_curve
from rotation_fitfunctions import parameterfit_bur, parameterfit_iso, parameterfit_NFW

H_0 = 100  # Hubble's Constant in units of h km/s/Mpc
c = 299792.458  # Speed of light in units of km/s

"""def combination_fit(sMass_density, sMass_density_err, r_band, vmap, ivar, \
                    map_mask, x0, y0, axis_ratio, phi, z, vsys_0, rho_h0, Rh_0, gal_ID):
    
    
    PARAMETERS
    
    sMass_density, sMass_density_err: masked numpy array
        Masked stellar mass density (log(Msun/spaxel^2)) and associated error map
    
    r_band: numpy array
        r_band flux 
    
    vmap: masked numpy array
        masked Halpha velocity map (km/s)
    
    ivar: masked numpy array
        masked inverse variance of velocity map
    
    map_mask: numpy array mask
        mask for the maps
     
    x0, y0: list
        x and y centers for each fit model (Iso, NFW, bur) (spaxel)
    
    axis_ratio: list 
        axis_ratio for each fit model (Iso, NFW, bur)
    
    phi: list 
        rotation angle EofN for each fit model (Iso, NFW, bur) (radians)
    
    z: float
        redshift of galaxy
        
    rho_h0: float
        intial guess for the central halo density (log(M_sun/pc^3)
     
    Rh_0: float
        intial guess for halo scale radius (kpc)
    
    vsys_0: float
        intial guess for systematic velocity (km/s)
    
    gal_ID: string
        Plate-IFU 
    
    RETURNS
    
    fit parameters for each halo model
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
    
    
    
    x0_iso, x0_NFW, x0_bur = x0
    y0_iso, y0_NFW, y0_bur = y0
    ba_iso, ba_NFW, ba_bur = axis_ratio
    phi_iso, phi_NFW, phi_bur = phi
    incl_iso, incl_NFW, incl_bur = np.arccos(np.sqrt((np.power(axis_ratio,2) - 0.2 ** 2) / (1 - 0.2 ** 2)))
    vsys_0iso, vsys_0NFW, vsys_0bur = vsys_0
    rho_h0_iso, rho_h0_NFW, rho_h0_bur = rho_h0
    Rh_0_iso, Rh_0_NFW, Rh_0_bur = Rh_0
    scale = (0.5 * z * c / H_0) * 1000 / 206265  # converting radii to kpc

    
    
    param_iso = [rho_h0_iso, Rh_0_iso, incl_iso, phi_iso, x0_iso, y0_iso, vsys_0iso]
    param_bur = [rho_h0_NFW, Rh_0_NFW, incl_bur, phi_bur, x0_bur, y0_bur, vsys_0NFW]
    param_NFW = [rho_h0_bur, Rh_0_bur, incl_NFW, phi_NFW, x0_NFW, y0_NFW, vsys_0bur]
    

    # Disk Mass fit
    if x0_iso == x0_NFW and y0_iso == y0_NFW and ba_iso == ba_NFW and phi_iso == phi_NFW:
        mass_data_table = calc_mass_curve(sMass_density, sMass_density_err, r_band, \
                                          map_mask, x0_iso, y0_iso, ba_iso, phi_iso, z, gal_ID)
        param_outputs = fit_mass_curve(mass_data_table, gal_ID, 'bulge')
        param_outputs_iso = param_outputs
        param_outputs_NFW = param_outputs
        param_outputs_bur = param_outputs
    else:
        mass_data_table_iso = calc_mass_curve(sMass_density, sMass_density_err, r_band, \
                                              map_mask, x0_iso, y0_iso, ba_iso, phi_iso, z, gal_ID)
        mass_data_table_NFW = calc_mass_curve(sMass_density, sMass_density_err, r_band, \
                                              map_mask, x0_NFW, y0_NFW, ba_NFW, phi_NFW, z, gal_ID)
        mass_data_table_bur = calc_mass_curve(sMass_density, sMass_density_err, r_band, \
                                              map_mask, x0_bur, y0_bur, ba_bur, phi_bur, z, gal_ID)
        param_outputs_iso = fit_mass_curve(mass_data_table_iso, gal_ID, 'bulge')
        param_outputs_NFW = fit_mass_curve(mass_data_table_NFW, gal_ID, 'bulge')
        param_outputs_bur = fit_mass_curve(mass_data_table_bur, gal_ID, 'bulge')

    ############################################################################

    # Total velocity fits
    best_fit_iso = parameterfit_iso(param_iso, param_outputs_iso['rho_bulge'], param_outputs_iso['R_bulge'], \
                                    param_outputs_iso['Sigma_disk'], param_outputs_iso['R_disk'], scale, \
                                    vmap.shape, vmap, ivar, map_mask)
    best_fit_NFW = parameterfit_NFW(param_NFW, param_outputs_NFW['rho_bulge'], param_outputs_NFW['R_bulge'], \
                                    param_outputs_NFW['Sigma_disk'], param_outputs_NFW['R_disk'], scale, \
                                    vmap.shape, vmap, ivar, map_mask)
    best_fit_bur = parameterfit_bur(param_bur, param_outputs_bur['rho_bulge'], param_outputs_bur['R_bulge'], \
                                    param_outputs_bur['Sigma_disk'], param_outputs_bur['R_disk'], scale, \
                                    vmap.shape, vmap, ivar, map_mask)

    iso_fit = [param_outputs_iso['rho_bulge'], param_outputs_iso['R_bulge'], param_outputs_iso['Sigma_disk'], \
               param_outputs_iso['R_disk'], best_fit_iso[0], best_fit_iso[1], best_fit_iso[2], \
               best_fit_iso[3], best_fit_iso[4], best_fit_iso[5], best_fit_iso[6]]
    NFW_fit = [param_outputs_NFW['rho_bulge'], param_outputs_NFW['R_bulge'], param_outputs_NFW['Sigma_disk'], \
               param_outputs_NFW['R_disk'], best_fit_NFW[0], best_fit_NFW[1], best_fit_NFW[2], \
               best_fit_NFW[3], best_fit_NFW[4], best_fit_NFW[5], best_fit_NFW[6]]

    bur_fit = [param_outputs_bur['rho_bulge'], param_outputs_bur['R_bulge'], param_outputs_bur['Sigma_disk'], \
               param_outputs_bur['R_disk'], best_fit_bur[0], best_fit_bur[1], best_fit_bur[2], \
               best_fit_bur[3], best_fit_bur[4], best_fit_bur[5], best_fit_bur[6]]
    ###################################################
    # Plotting velocity fits
    RC.plot_diagnostic_panel(gal_ID, vmap.shape, scale, iso_fit, NFW_fit, bur_fit, map_mask, vmap, ivar)
    
    return iso_fit, NFW_fit, bur_fit
"""
def combination_fit(sMass_density, sMass_density_err, r_band, vmap, ivar, map_mask, para_guesses, z, gal_ID, fit_function):
    
    
    rho_h0, Rh_0, axis_ratio, phi, x0, y0,  vsys_0 = para_guesses
    incl = np.arccos(np.sqrt((axis_ratio **2  - 0.2 ** 2) / (1 - 0.2 ** 2)))
    scale = (0.5 * z * c / H_0) * 1000 / 206265
    param = rho_h0, Rh_0, incl, phi, x0, y0, vsys_0
    shape = vmap.shape
##disk fit
    mass_data_table = calc_mass_curve(sMass_density, sMass_density_err, r_band, \
                                          map_mask, x0, y0, axis_ratio, phi, z, gal_ID)
    param_outputs = fit_mass_curve(mass_data_table, gal_ID, 'bulge')
    
#total fit
    if fit_function == 'Isothermal':
        best_fit = parameterfit_iso(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                       param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                       shape, vmap, ivar, map_mask)
    elif fit_function == 'NFW':
        best_fit = parameterfit_NFW(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                       param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                       shape, vmap, ivar, map_mask)
   
    elif fit_function == 'Burkert':
        best_fit = parameterfit_bur(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                       param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                       shape, vmap, ivar, map_mask)
    else:
        print("Fit function not known")
    best_fit_values = [param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                           param_outputs['Sigma_disk'], param_outputs['R_disk'], \
                           best_fit[0], best_fit[1], best_fit[2], best_fit[3], \
                           best_fit[4],best_fit[5],best_fit[6]]
#Plot
    RC.plot_rot_curve(vmap, ivar, best_fit_values, scale, gal_ID, fit_function)
    return best_fit_values



