########## imports
import numpy as np
import numpy.ma as ma
from astropy.table import Table
import matplotlib.pyplot as plt
from disk_mass import calc_mass_curve, fit_mass_curve
from rotation_fitfunctions import parameterfit_bur, parameterfit_iso, parameterfit_NFW,find_phi, find_axis_ratio, chi2
from DRP_rotation_curve import extract_data, extract_Pipe3d_data
from disk_mass_functions import chi2_mass
from disk_mass_plotting_functions import plot_fitted_disk_rot_curve
from RC_plotting_functions import plot_totfit_panel

##################################################
#Constants
H_0 = 100  # Hubble's Constant in units of h km/s/Mpc
c = 299792.458  # Speed of light in units of km/s
#####################################################

FILE_IDS = ['7443-12705','8997-9102','8985-9102']
RUN_ALL_GALAXIES = False
fit_function = "Isothermal"

#for bluehive
MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
MASS_MAP_FOLDER = MANGA_FOLDER + 'pipe3d/'
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/' + 'drpall-v3_1_1.fits'

##############################################################
#Read in the galaxy information

DRP_table = Table.read( DRP_FILENAME, format='fits')
print(DRP_table['plateifu'][0])

DRP_index = {}

for i in range(len(DRP_table)):
    galaxy_ID = DRP_table['plateifu'][i]

    DRP_index[galaxy_ID] = i

##################################################################
#Set up output table
#fit_table = Table()

#####################################################################

if RUN_ALL_GALAXIES:
    
    N_files = len(master_table)
    FILE_IDS = list(master_index.keys())

else:

    N_files = len(FILE_IDS)


for gal_ID in FILE_IDS:
    #get data and initial parameters
    
    maps = extract_data(VEL_MAP_FOLDER,gal_ID,['Ha_vel', 'r_band', 'Ha_flux', 'Ha_sigma'])
    sMass_density, sMass_density_err = extract_Pipe3d_data(MASS_MAP_FOLDER, gal_ID)
    i_DRP = DRP_index[gal_ID]
    axis_ratio = DRP_table['nsa_elpetro_ba'][i_DRP]
    phi = DRP_table['nsa_elpetro_phi'][i_DRP]
    z = DRP_table['nsa_z'][i_DRP]
    SN_map = maps['Ha_flux'] * np.sqrt(maps['Ha_flux_ivar'])
    vmap_mask = maps['Ha_vel_mask'] + (SN_map < 5)
    sM_mask = maps['Ha_vel_mask']
    maps['vmasked'] = ma.array(maps['Ha_vel'], mask=vmap_mask)
    maps['ivarmasked'] = ma.array(maps['Ha_vel_ivar'], mask=vmap_mask)
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
            best_fit_values = None
            fit = None
            break;
        #total fit
        print("Fitting velocity map")
        if fit_function == 'Isothermal':
            best_fit,uncertainties = parameterfit_iso(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                           param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                           shape, maps['vmasked'], maps['ivarmasked'], vmap_mask)
        elif fit_function == 'NFW':
            best_fit,uncertainties = parameterfit_NFW(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                           param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                           shape, maps['vmasked'], maps['ivarmasked'], vmap_mask)
       
        elif fit_function == 'Burkert':
            best_fit,uncertainties = parameterfit_bur(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
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
    
    if fit != None:
        fit = [param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                           param_outputs['Sigma_disk'], param_outputs['R_disk'], \
                           best_fit[0][0], best_fit[0][1], best_fit[0][2], best_fit[0][3], \
                           best_fit[0][4],best_fit[0][5],best_fit[0][6]]
    
        sM_chi2 = chi2_mass(fit[0:4], mass_data_table['radius'], mass_data_table['star_vel'], mass_data_table['star_vel_err'])
        plot_fitted_disk_rot_curve(gal_ID, \
                                       mass_data_table, \
                                       param_outputs, \
                                       sM_chi2, \
                                       'bulge', IMAGE_DIR = "/gpfs/fs1/home/lstroud3/Documents/RotationCurve/2D_RC",IMAGE_FORMAT='png')
        plot_totfit_panel (gal_ID, shape, scale, fit, fit_function, vmap_mask, maps['vmasked'],\
                              maps['ivarmasked'], mass_data_table, param_outputs)
    