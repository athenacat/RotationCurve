import numpy as np
import numpy.ma as ma
import RC_plotting_functions as RC
from disk_mass import calc_mass_curve, fit_mass_curve
from rotation_fitfunctions import parameterfit_bur, parameterfit_iso, parameterfit_NFW, calc_hessian, find_incl

H_0 = 100  # Hubble's Constant in units of h km/s/Mpc
c = 299792.458  # Speed of light in units of km/s


def combination_fit(sMass_density, sMass_density_err, r_band, vmap, ivar, vmap_mask, sM_mask, para_guesses, z, gal_ID, fit_function):
    
    
    rho_h0, Rh_0, axis_ratio, phi, x0, y0,  vsys_0 = para_guesses
    incl = find_incl(axis_ratio)
    scale = (0.5 * z * c / H_0) * 1000 / 206265
    param = rho_h0, Rh_0, incl, phi, x0, y0, vsys_0
    shape = vmap.shape
##disk fit
    print("Fitting disk")
    mass_data_table = calc_mass_curve(sMass_density, sMass_density_err, r_band, \
                                          sM_mask, x0, y0, axis_ratio, phi, z, gal_ID)
    param_outputs = fit_mass_curve(mass_data_table, gal_ID, 'bulge')
    print(param_outputs)
    
#total fit
    print("Fitting velocity map")
    if fit_function == 'Isothermal':
        best_fit = parameterfit_iso(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                       param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                       shape, vmap, ivar, vmap_mask, gal_ID)
    elif fit_function == 'NFW':
        best_fit = parameterfit_NFW(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                       param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                       shape, vmap, ivar, vmap_mask, gal_ID)
   
    elif fit_function == 'Burkert':
        best_fit = parameterfit_bur(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                       param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                       shape, vmap, ivar, vmap_mask, gal_ID)
    else:
        print("Fit function not known")
    
    uncertainties = calc_hessian(best_fit, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                 param_outputs['Sigma_disk'], param_outputs['R_disk'], fit_function,\
                 scale, shape, vmap, ivar, vmap_mask, gal_ID)
                     
    
    best_fit_values = [param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                           param_outputs['Sigma_disk'], param_outputs['R_disk'], \
                           best_fit[0], best_fit[1], best_fit[2], best_fit[3], \
                           best_fit[4],best_fit[5],best_fit[6]]
    
    
    
    return best_fit_values, uncertainties, mass_data_table, param_outputs


    

