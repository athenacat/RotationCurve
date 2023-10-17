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
#batchnum =0 #int(sys.argv[1])
RUN_ALL_GALAXIES = False
fit_function = sys.argv[1] #"Burkert"
smoothness_max = 2.0

#########################################################

#for bluehive
IMAGE_DIR = '/scratch/lstroud3/RotationCurves/Images/'
OUT_FILE_FOLDER = IMAGE_DIR + fit_function + "/"
"""
# Create directory if it does not already exist
if not os.path.isdir( IMAGE_DIR):
    os.makedirs( IMAGE_DIR)
if not os.path.isdir(OUT_FILE_FOLDER):
    os.makedirs(OUT_FILE_FOLDER)
    """
MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
MASS_MAP_FOLDER = MANGA_FOLDER + 'pipe3d/'
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/' + 'drpall-v3_1_1.fits'
MORPH_FILE = '/home/lstroud3/Documents/manga_visual_morpho-2.0.1.fits'


##############################################################
#reading in fixed galaxies

failed_object_table = Table.read('/scratch/lstroud3/RotationCurves/failed_objects_table.fits')



####################################################3

#Read in the galaxy information

DRP_table = Table.read( DRP_FILENAME, format='fits')
DRP_index = {}
index = []
failed_centers = failed_object_table[failed_object_table['visual code']==1]
FILE_IDS = failed_centers['plateifu']

print(FILE_IDS)                       
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
    
    maps = extract_data(VEL_MAP_FOLDER,gal_ID,['Ha_vel', 'r_band', 'Ha_flux', 'Ha_sigma'])
    if gal_ID != '11828-1902':
        sMass_density, sMass_density_err = extract_Pipe3d_data(MASS_MAP_FOLDER, gal_ID)
        else:
            fit_flag = -10
    if maps == None or fit_flag = -10:
            fit_flag = -10
     

     
    if fit_flag == 0:
        SN_map = maps['Ha_flux'] * np.sqrt(maps['Ha_flux_ivar'])
        vmap_mask = maps['Ha_vel_mask'] + (SN_map < 5)
        sM_mask = maps['Ha_vel_mask']
        maps['vmasked'] = ma.array(maps['Ha_vel'], mask=vmap_mask)
        maps['ivarmasked'] = ma.array(maps['Ha_vel_ivar'], mask=vmap_mask)
        maps['rbandmasked'] = ma.array(maps['r_band'], mask = maps['Ha_vel_mask'])
        z = DRP_table['nsa_z'][i_DRP]
        print("Getting parameters:",flush=True)
        axis_ratio = DRP_table['nsa_sersic_ba'][i_DRP]
        incl = find_incl(axis_ratio)
        shape = maps['vmasked'].shape
        scale = (0.5 * z * c / H_0) * 1000 / 206265  # kpc
            
        if gal_ID in ['10220-12703','10516-12705','10520-12705','12066-12703',\
                         '11012-12703','8239-12703','8941-12703','8717-12705',\
                          '10841-12705','11018-12701','11743-12701','11751-12702',\
                          '11824-12702','11832-12702','11865-12705','12651-12701',\
                          '8088-12704','8438-12705','8711-12701','8950-12705','9037-9102'\
                         '9498-12703',]:
            center = (37,37)
                
        elif gal_ID in ['8466-12705','11021-12702']:
            center = (37,42)
                
        elif gal_ID in ['10222-6102','8133-6102','8252-6103']:
            center = (27,27)

        elif gal_ID == '11830-3704':
            center = (20,19)         
            
        elif gal_ID in ['9879-6101','10845-6101','11758-6103','9872-6103']:
            center = (30,27)      

        elif gal_ID in ['7443-3701','8335-3704','9514-3702','8318-3702','8657-3702','7815-3704','8341-3702']:
            center = (22,22)  

        elif gal_ID in ['8728-6104','12488-6102','8551-6104','8570-6104']:
            center = (25,25)
        elif gal_ID in ['10223-12704']:
            center = (41,31)
        elif gal_ID == '11945-9101':
            center = (31,31)
            
        elif gal_ID == '12066-12703':
            center = (37,30)
                         
        elif gal_ID == '10495-12704':
            center = (39,36)
            
        elif gal_ID in ['10515-3703','8651-3702']:
            center = (20,25)
        elif gal_ID in ['11021-12702','9891-12704']:
            center = (35,40)
        elif gal_ID in ['11823-12702','11947-12703']:
            center = (40,40)
        elif gal_ID == '8240-12705':
            center = (45,45)
        elif gal_ID in ['8613-12701','9890-12705']:
            center = (40,37)
        elif gal_ID == '8626-1902':
            center = (17,17)
        elif gal_ID == '9046-12705':
            center = (30,35)
        else:
            center = np.unravel_index(ma.argmax(maps['rbandmasked']), shape)
        x0 = center[0]
        y0 = center[1]
        hoh = -1.5
        Rh = 10
        vsys = 0
        
        phi_guesses = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170]
        
    ################################################
        for phi in phi_guesses:
            fitting_code = 0
            phi_guess = find_phi(phi,center,maps['vmasked'])
            param = [rhoh,Rh,incl,phi_guess,x0,y0,vsys]
            fit = []
            compare_table = Table(names=('rhob','Rb','sigd','Rd','rhoh','Rh','incl','phi','x0','y0','vsys','chi2'))
            no_fits = True
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
                    
                fit = [param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                       param_outputs['Sigma_disk'], param_outputs['R_disk'], \
                                       best_fit[0], best_fit[1], best_fit[2], best_fit[3], \
                                       best_fit[4],best_fit[5],best_fit[6]]
            
                sM_chi2 = chi2_mass(fit[0:4], mass_data_table['radius'], mass_data_table['star_vel'], mass_data_table['star_vel_err'])               
                c2 = chi2(maps['vmasked'], maps['ivarmasked'],vmap_mask,shape,scale,fit,fit_function)[1]
                compare_table.add_row((fit[0],fit[1],fit[2],fit[3],fit[4],fit[5],fit[6],fit[7],fit[8],fit[9],fit[10],c2))
                no_fit = False
        
        if no_fit = False:
                compare.sort('chi2',reverse=True)
                
                
                

                uncertainties = calc_hessian(best_fit, param_outputs['rho_bulge'], param_outputs['R_bulge'],\
                                         param_outputs['Sigma_disk'], param_outputs['R_disk'], fit_function, scale,\
                                         shape, maps['vmasked'], maps['ivarmasked'], vmap_mask, gal_ID)
                
                
                
                plot_fitted_disk_rot_curve(gal_ID, \
                                               mass_data_table, \
                                               param_outputs, \
                                               sM_chi2, \
                                               'bulge', IMAGE_DIR = IMAGE_DIR)
                plot_rot_curve (maps['vmasked'], maps['ivarmasked'], fit, scale, gal_ID, fit_function, \
                                       IMAGE_DIR = IMAGE_DIR, IMAGE_FORMAT='png')


        out_table['fit flag'][i_gal] = fit_flag
        print(fit_flag, flush=True)
    print(gal_ID," Time: ",datetime.datetime.now() - TIME , flush=True)

    if True: #galcount%20 == 0 or N_files < 20:
        out_table.write(OUT_FILE_FOLDER+"centerfix"+'.fits',format='fits',overwrite = True)
print('Runtime:',datetime.datetime.now() - START, flush = True)