########## imports
import numpy as np
import numpy.ma as ma
from astropy.table import Table
import os.path, warnings
from disk_mass import calc_mass_curve, fit_mass_curve
from rotation_fitfunctions import parameterfit_bur, parameterfit_iso, parameterfit_NFW,find_phi, find_axis_ratio, chi2, calc_hessian
from DRP_rotation_curve import extract_data, extract_Pipe3d_data
from disk_mass_functions import chi2_mass
from disk_mass_plotting_functions import plot_fitted_disk_rot_curve
from RC_plotting_functions import plot_totfit_panel

import sys
sys.path.insert(1,'/main/')
from mapSmoothness_functions import how_smooth

warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', RuntimeWarning)

##################################################
#Constants
H_0 = 100  # Hubble's Constant in units of h km/s/Mpc
c = 299792.458  # Speed of light in units of km/s
#####################################################

FILE_IDS = ['7443-12705','8997-9102','8985-9102']
RUN_ALL_GALAXIES = False
fit_function = "Isothermal"
smoothness_max = 2.0

#########################################################


IMAGE_DIR = '/scratch/lstroud3/RotationCurves/Images/'
# Create directory if it does not already exist
if not os.path.isdir( IMAGE_DIR):
    os.makedirs( IMAGE_DIR)

#for bluehive
MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
MASS_MAP_FOLDER = MANGA_FOLDER + 'pipe3d/'
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/' + 'drpall-v3_1_1.fits'
MORPH_FILE = '/home/lstroud3/Documents/RotationCurve/2D_RC/manga_visual_morpho-2.0.1.fits'
##############################################################
#Read in the galaxy information

DRP_table = Table.read( DRP_FILENAME, format='fits')
DRP_index = {}

for i in range(len(DRP_table)):
    galaxy_ID = DRP_table['plateifu'][i]

    DRP_index[galaxy_ID] = i


Morph_table = Table.read(MORPH_FILE, format='fits')
Morph_index = {}

for i in range(len(Morph_table)):
    galaxy_ID = Morph_table['plateifu'][i]
    Morph_index[galaxy_ID] = i


    
##################################################################
#initialize counts for cut galaxies
num_masked = 0 # Number of completely masked galaxies
num_not_smooth = 0 # Number of galaxies which do not have smooth velocity maps
num_wrong_ttype = 0 #Number of galaxies with ttype < 0
num_tidal = 0 #Number of galaxies with tidal debris




#####################################################################
if RUN_ALL_GALAXIES:
    
    N_files = len(master_table)
    FILE_IDS = list(master_index.keys())

else:

    N_files = len(FILE_IDS)
#####################################################################
    
    
for gal_ID in FILE_IDS:
    if (DRP_table['mngtarg1'][i_DRP] > 0) or (gal_ID in ['9037-9102']):
        #read in data
        maps = extract_data(VEL_MAP_FOLDER,gal_ID,['Ha_vel', 'r_band', 'Ha_flux', 'Ha_sigma'])
        sMass_density, sMass_density_err = extract_Pipe3d_data(MASS_MAP_FOLDER, gal_ID)
        i_DRP = DRP_index[gal_ID]
        i_morph = Morph_index[gal_ID]
        SN_map = maps['Ha_flux'] * np.sqrt(maps['Ha_flux_ivar'])
        vmap_mask = maps['Ha_vel_mask'] + (SN_map < 5)
        sM_mask = maps['Ha_vel_mask']
        maps['vmasked'] = ma.array(maps['Ha_vel'], mask=vmap_mask)
        maps['ivarmasked'] = ma.array(maps['Ha_vel_ivar'], mask=vmap_mask)
        
        
        ###checking if galaxy meets criteria to fit
        can_fit = True
        map_smoothness = how_smooth(maps['Ha_vel'], maps['Ha_vel_mask'])
        
        if map_smoothness > smoothness_max:
            can_fit = False
            print('Not smooth:'+gal_ID)
            num_not_smooth += 1
        
        elif Mdata['TTYPE'][i_morph] <= 0:
            can_fit = False
            num_wrong_ttype += 1
            print('Not ttype:'+gal_ID)
        elif Mdata['TIDAL'][i_morph] != 0:
            can_fit = False
            print('tidal:'+gal_ID)
            num_tidal += 1
        
        elif (maps['vmasked'].count() / (maps['Ha_vel'].shape[0]*maps['Ha_vel'].shape[1])) < 0.05:
            can_fit = False
            print('masked:'+gal_ID)
            num_masked += 1
        
            
        
        if can_fit:
            axis_ratio = DRP_table['nsa_elpetro_ba'][i_DRP]
            phi = DRP_table['nsa_elpetro_phi'][i_DRP]
            z = DRP_table['nsa_z'][i_DRP]
            shape = maps['vmasked'].shape
            scale = (0.5 * z * c / H_0) * 1000 / 206265  # kpc
            center = np.unravel_index(ma.argmax(maps['r_band']), shape)
            x0 = center[0]
            y0 = center[1]
            phi = find_phi(center, phi, maps['vmasked'])
            rhoh = -1.5
            Rh = 10
            vsys = 0
            param = [rhoh,Rh,axis_ratio,phi,x0,y0,vsys]
    
    ################################################
            for i in range(50):
                ##disk fit
                print("Fitting disk")
                mass_data_table = calc_mass_curve(sMass_density, sMass_density_err, maps['r_band'], \
                                                  sM_mask, x0, y0, axis_ratio, phi, z, gal_ID)
                
                param_outputs = fit_mass_curve(mass_data_table, gal_ID, 'bulge')
                if param_outputs == None:
                    fit = np.nan * np.ones(11)
                    break;
                #total fit
                print("Fitting velocity map")
                if fit_function == 'Isothermal':
                    best_fit = parameterfit_iso(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                                   param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                                   shape, maps['vmasked'], maps['ivarmasked'], vmap_mask)
                elif fit_function == 'NFW':
                    best_fit = parameterfit_NFW(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                                   param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                                   shape, maps['vmasked'], maps['ivarmasked'], vmap_mask)
               
                elif fit_function == 'Burkert':
                    best_fit = parameterfit_bur(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                                   param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                                   shape, maps['vmasked'], maps['ivarmasked'], vmap_mask)
                else:
                    print("Fit function not known")
                
                #check if fit is close enough to past iteration
                if np.abs(best_fit[2]-param[2])<0.001 and np.abs(best_fit[4]-param[4])<1 and np.abs(best_fit[3]-param[3])<0.0017 \
                        and np.abs(best_fit[5]-param[5])<1:
                    print("Iteration Converged")
                    break
                
                param = best_fit
                
            if fit != np.nan * np.ones(11):
                
                #calc_hessian(fit, rhob, Rb, SigD, Rd, fit_function, scale, shape, vmap, ivar, mask, gal_ID)
                uncertainties = calc_hessian(best_fit, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                                   param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                                   shape, maps['vmasked'], maps['ivarmasked'], vmap_mask, gal_ID)
                fit = [param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                   param_outputs['Sigma_disk'], param_outputs['R_disk'], \
                                   best_fit[0], best_fit[1], best_fit[2], best_fit[3], \
                                   best_fit[4],best_fit[5],best_fit[6]]
            
                sM_chi2 = chi2_mass(fit[0:4], mass_data_table['radius'], mass_data_table['star_vel'], mass_data_table['star_vel_err'])
                plot_fitted_disk_rot_curve(gal_ID, \
                                               mass_data_table, \
                                               param_outputs, \
                                               sM_chi2, \
                                               'bulge', IMAGE_DIR = IMAGE_DIR)
                plot_totfit_panel (gal_ID, shape, scale, fit, fit_function, vmap_mask, maps['vmasked'],\
                                      maps['ivarmasked'], mass_data_table, param_outputs)
             