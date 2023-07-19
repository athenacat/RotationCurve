########## imports
import datetime
START = datetime.datetime.now()
import numpy as np
import numpy.ma as ma
from astropy.table import Table
import os.path, warnings
from disk_mass import calc_mass_curve, fit_mass_curve
from rotation_fitfunctions import parameterfit_bur, parameterfit_iso, parameterfit_NFW,find_phi, find_axis_ratio, find_incl, chi2, calc_hessian
from DRP_rotation_curve import extract_data, extract_Pipe3d_data
from disk_mass_functions import chi2_mass
from disk_mass_plotting_functions import plot_fitted_disk_rot_curve
from RC_plotting_functions import plot_rot_curve

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

#FILE_IDS = ['10219-9102','10220-12701']  #use if directly naming files
batchnum = int(sys.argv[1])
RUN_ALL_GALAXIES = False
fit_function = "Burkert"
smoothness_max = 2.0

#########################################################

#for bluehive
IMAGE_DIR = '/scratch/lstroud3/RotationCurves/Images/'
OUT_FILE_FOLDER = IMAGE_DIR + fit_function + "/"
# Create directory if it does not already exist
if not os.path.isdir( IMAGE_DIR):
    os.makedirs( IMAGE_DIR)
if not os.path.isdir(OUT_FILE_FOLDER):
    os.makedirs(OUT_FILE_FOLDER)
MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
MASS_MAP_FOLDER = MANGA_FOLDER + 'pipe3d/'
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/' + 'drpall-v3_1_1.fits'
MORPH_FILE = '/home/lstroud3/Documents/manga_visual_morpho-2.0.1.fits'


##############################################################
#Read in the galaxy information

DRP_table = Table.read( DRP_FILENAME, format='fits')
DRP_index = {}
index = []
FILE_IDS = DRP_table['plateifu'][(batchnum*1000):((batchnum+1)*1000)] #use to take rows of the DRPall
for i in range(len(DRP_table)):
    galaxy_ID = DRP_table['plateifu'][i]
    DRP_index[galaxy_ID] = i
    if RUN_ALL_GALAXIES or galaxy_ID in FILE_IDS:
        index.append(i)

out_table = Table(DRP_table[index])        

Morph_table = Table.read(MORPH_FILE, format='fits')
Morph_index = {}

for i in range(len(Morph_table)):
    galaxy_ID = Morph_table['plateifu'][i].strip()
    Morph_index[galaxy_ID] = i
    
out_index = {}
if not RUN_ALL_GALAXIES:    
    for i in range(len(out_table)):
            galaxy_ID = out_table['plateifu'][i]
            out_index[galaxy_ID] = i
else:
    out_index = DRP_index
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
###adding columns to the output table
out_table['rho bulge'] = np.nan
out_table['R bulge'] = np.nan
out_table['sigma disk'] = np.nan
out_table['R disk'] = np.nan
out_table['rho halo'] = np.nan
out_table['R halo'] = np.nan
out_table['incl'] = np.nan
out_table['phi'] = np.nan
out_table['x0'] = np.nan
out_table['y0'] = np.nan
out_table['vsys'] = np.nan
out_table['rho bulge err'] = np.nan
out_table['R bulge err'] = np.nan
out_table['sigma disk err'] = np.nan
out_table['R disk err'] = np.nan
out_table['rho halo err'] = np.nan
out_table['R halo err'] = np.nan
out_table['incl err'] = np.nan
out_table['phi err'] = np.nan
out_table['x0 err'] = np.nan
out_table['y0 err'] = np.nan
out_table['vsys err'] = np.nan
out_table['chi2'] = np.nan
out_table['fit flag'] = np.nan
#####################################################################
galcount = 0 
for gal_ID in FILE_IDS:
    galcount += 1
    TIME = datetime.datetime.now()
    print(gal_ID, flush = True)
    i_DRP = DRP_index[gal_ID]
    i_gal = out_index[gal_ID]
    if (DRP_table['mngtarg1'][i_DRP] > 0) or (gal_ID in ['9037-9102']):
        i_morph = Morph_index[gal_ID]
        maps = extract_data(VEL_MAP_FOLDER,gal_ID,['Ha_vel', 'r_band', 'Ha_flux', 'Ha_sigma'])
        sMass_density, sMass_density_err = extract_Pipe3d_data(MASS_MAP_FOLDER, gal_ID)
        SN_map = maps['Ha_flux'] * np.sqrt(maps['Ha_flux_ivar'])
        vmap_mask = maps['Ha_vel_mask'] + (SN_map < 5)
        sM_mask = maps['Ha_vel_mask']
        maps['vmasked'] = ma.array(maps['Ha_vel'], mask=vmap_mask)
        maps['ivarmasked'] = ma.array(maps['Ha_vel_ivar'], mask=vmap_mask)
        z = DRP_table['nsa_z'][i_DRP]
        
        ###checking if galaxy meets criteria to fit
        fit_flag = 0
        map_smoothness = how_smooth(maps['Ha_vel'], maps['Ha_vel_mask'])
        
        if map_smoothness > smoothness_max:
            fit_flag = -1
            print('Not smooth:'+gal_ID, flush=True)
            num_not_smooth += 1
        
        elif Morph_table['TType'][i_morph] <= 0:
            fit_flag = -2
            num_wrong_ttype += 1
            print('Not ttype:'+gal_ID, flush=True)
        elif Morph_table['Tidal'][i_morph] != 0:
            fit_flag = -3
            print('tidal:'+gal_ID, flush=True)
            num_tidal += 1
        
        elif (maps['vmasked'].count() / (maps['Ha_vel'].shape[0]*maps['Ha_vel'].shape[1])) < 0.05:
            fit_flag = -4
            print('masked:'+gal_ID, flush=True)
            num_masked += 1
        
        
        else:
            print("Getting parameters:",flush=True)
            axis_ratio = DRP_table['nsa_sersic_ba'][i_DRP]
            incl = find_incl(axis_ratio)
            phi = DRP_table['nsa_elpetro_phi'][i_DRP]
            shape = maps['vmasked'].shape
            scale = (0.5 * z * c / H_0) * 1000 / 206265  # kpc
            
            if gal_ID == '10220-12703':
                center = (38,37)
            else:
                center = np.unravel_index(ma.argmax(maps['r_band']), shape)
            x0 = center[0]
            y0 = center[1]
            phi = find_phi(center, phi, maps['vmasked'])
            rhoh = -1.5
            Rh = 10
            vsys = 0
            param = [rhoh,Rh,incl,phi,x0,y0,vsys]
    
    ################################################
            fit = []
            for i in range(50):
                ##disk fit
                #print("Fitting disk")
                
                rhoh,Rh,incl,phi,x0,y0,vsys = param
                axis_ratio = find_axis_ratio(incl)
                
                mass_data_table = calc_mass_curve(sMass_density, sMass_density_err, maps['r_band'], \
                                                  sM_mask, x0, y0, axis_ratio, phi, z, gal_ID)
                if len(mass_data_table)<10:
                    fit_flag = -5
                    break;
                param_outputs = fit_mass_curve(mass_data_table, gal_ID, 'bulge')
                
                if param_outputs == None:
                    fit_flag = -5
                    break;
                #total fit
                print("Fitting velocity map", flush=True)
                if fit_function == 'Isothermal':
                    best_fit = parameterfit_iso(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                                   param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                                   shape, maps['vmasked'], maps['ivarmasked'], vmap_mask, gal_ID)
                elif fit_function == 'NFW':
                    best_fit = parameterfit_NFW(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                                   param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                                   shape, maps['vmasked'], maps['ivarmasked'], vmap_mask, gal_ID)
               
                elif fit_function == 'Burkert':
                    best_fit = parameterfit_bur(param, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                                   param_outputs['Sigma_disk'], param_outputs['R_disk'], scale,\
                                                   shape, maps['vmasked'], maps['ivarmasked'], vmap_mask, gal_ID)
                else:
                    print("Fit function not known", flush=True)
                
                if best_fit is None:
                    fit_flag = -6
                    break;
                
                
                #check if fit is close enough to past iteration
                if np.abs(best_fit[2]-param[2])<0.001 and np.abs(best_fit[4]-param[4])<1 and np.abs(best_fit[3]-param[3])<0.0017 \
                        and np.abs(best_fit[5]-param[5])<1 and np.abs(best_fit[6]-param[6])<0.01:
                    fit_flag = i
                    print("Converged", flush=True)
                    break
                
                param = best_fit
                
                
                
            if fit_flag >= 0:
                uncertainties = calc_hessian(best_fit, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                         param_outputs['Sigma_disk'], param_outputs['R_disk'], fit_function, scale,\
                                         shape, maps['vmasked'], maps['ivarmasked'], vmap_mask, gal_ID)
                fit = [param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                   param_outputs['Sigma_disk'], param_outputs['R_disk'], \
                                   best_fit[0], best_fit[1], best_fit[2], best_fit[3], \
                                   best_fit[4],best_fit[5],best_fit[6]]
            
                sM_chi2 = chi2_mass(fit[0:4], mass_data_table['radius'], mass_data_table['star_vel'], mass_data_table['star_vel_err'])
                
                out_table['rho bulge'][i_gal] = param_outputs['rho_bulge']
                out_table['R bulge'][i_gal] = param_outputs['R_bulge']
                out_table['sigma disk'][i_gal] = param_outputs['Sigma_disk']
                out_table['R disk'][i_gal] = param_outputs['R_disk']
                out_table['rho bulge err'] = param_outputs['rho_bulge_err']
                out_table['R bulge err'] = param_outputs['R_bulge_err']
                out_table['sigma disk err'] =  param_outputs['Sigma_disk_err']
                out_table['R disk err'] = param_outputs['R_disk_err']
                
                

                out_table['rho halo'][i_gal], out_table['R halo'][i_gal], out_table['incl'][i_gal], out_table['phi'][i_gal], \
                out_table['x0'][i_gal], out_table['y0'][i_gal], out_table['vsys'][i_gal] = best_fit
              
                
                out_table['rho halo err'][i_gal], out_table['R halo err'][i_gal], out_table['incl err'][i_gal], \
                out_table['phi err'][i_gal], out_table['x0 err'][i_gal], out_table['y0 err'][i_gal], \
                out_table['vsys err'][i_gal] = uncertainties
                out_table['chi2'][i_gal] = chi2(maps['vmasked'], maps['ivarmasked'],vmap_mask,shape,scale,fit,fit_function)[1]
                
                
                
                plot_fitted_disk_rot_curve(gal_ID, \
                                               mass_data_table, \
                                               param_outputs, \
                                               sM_chi2, \
                                               'bulge', IMAGE_DIR = IMAGE_DIR)
                plot_rot_curve (maps['vmasked'], maps['ivarmasked'], fit, scale, gal_ID, fit_function, \
                                       IMAGE_DIR = IMAGE_DIR, IMAGE_FORMAT='png')


        out_table['fit flag'][i_gal] = fit_flag
    print(gal_ID," Time: ",datetime.datetime.now() - TIME , flush=True)
    print(fit_flag, flush=True)
    if galcount%20 == 0:
        out_table.write(OUT_FILE_FOLDER+str(batchnum+1),format='fits',overwrite = True)
print('Runtime:',datetime.datetime.now() - START, flush = True)