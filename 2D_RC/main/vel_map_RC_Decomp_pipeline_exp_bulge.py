################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
from astropy.table import Table, QTable
import csv


import time


# Import functions from other .py files
from Velocity_Map_Functions import find_phi

from RC_2D_Fit_Functions_old import Galaxy_Data, \
                                Galaxy_Fitting_iso,\
                                Galaxy_Fitting_NFW, \
                                Galaxy_Fitting_bur, \
                                Hessian_Calculation_Isothermal,\
                                Hessian_Calculation_NFW,\
                                Hessian_Calculation_Burket,\
                                Plotting_Isothermal,\
                                Plotting_NFW,\
                                Plotting_Burket,\
                                getTidal,\
                                deproject_spaxel,\
                                plot_rot_curve,\
                                plot_diagnostic_panel

from Velocity_Map_Functions_cython import rot_incl_iso,\
                                          rot_incl_NFW, \
                                          rot_incl_bur

from mapSmoothness_functions import how_smooth

from os import path

import matplotlib.pyplot as plt
################################################################################




################################################################################
# Physics Constants
#-------------------------------------------------------------------------------
c = 3E5 # km * s ^1
h = 1 # reduced hubble constant
H_0 =  100 * h # km * s^-1 * Mpc^-1
q0 = 0.2 # minimum inclination value
################################################################################




################################################################################
# Used files
#-------------------------------------------------------------------------------
MANGA_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
#MANGA_FOLDER = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
#MANGA_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'

DRP_FILENAME = MANGA_FOLDER + 'redux/v2_4_3/drpall-v2_4_3.fits'

VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'

MORPH_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/2D_RC/'
#MORPH_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/morphology/manga_visual_morpho/1.0.1/'


DTable =  Table.read(DRP_FILENAME, format='fits')

#MTable =  Table.read(MORPH_file, format='fits')

DRP_index = {}

for i in range(len(DTable)):
    gal_ID = DTable['plateifu'][i]

    DRP_index[gal_ID] = i
################################################################################





################################################################################
# Get the Mass of stars & redshifts & angular resolution of r50
#-------------------------------------------------------------------------------
m = DTable['nsa_elpetro_mass']
rat = DTable['nsa_elpetro_ba']
phi = DTable['nsa_elpetro_phi']
z = DTable['nsa_z']
r50_ang = DTable['nsa_elpetro_th50_r']
################################################################################





################################################################################
# Obtaining information for MaNGA galaxies
#-------------------------------------------------------------------------------
galaxy_ID = []

plate = ['7443','7495','7815']
IFU = ['1901','1902','3701','3702','3703','3704','6101','6102','6103','6104','9101','9102','12701','12702','12703','12704','12705']
#plate = ['7495']
#IFU = ['12704']

for i in range(len(plate)):
    for j in range(len(IFU)):
        galaxy_ID.append(plate[i] + '-' + IFU[j])

# Isothermal
#c_iso = open('iso_exp.csv','w')
#writer_iso = csv.writer(c_iso)
#writer_iso.writerow(['galaxy_ID', 'A', 'Vin', 'SigD', 'Rd', 'rho0_h', 'Rh', 'incl', 'phi', 'x_cen', 'y_cen','Vsys','chi2'])
c_iso = Table()
c_iso['galaxy_ID'] = galaxy_ID
c_iso['A'] = np.nan
c_iso['Vin'] = np.nan
c_iso['SigD'] = np.nan
c_iso['Rd'] = np.nan
c_iso['rho0_h'] = np.nan
c_iso['Rh'] = np.nan
c_iso['incl'] = np.nan
c_iso['phi'] = np.nan
c_iso['x_cen'] = np.nan
c_iso['y_cen'] = np.nan
c_iso['Vsys'] = np.nan
c_iso['chi2'] = np.nan

# NFW
#c_nfw = open('nfw_exp.csv','w')
#writer_nfw = csv.writer(c_nfw)
#writer_nfw.writerow(['galaxy_ID', 'A', 'Vin', 'SigD', 'Rd', 'rho0_h', 'Rh', 'incl', 'phi', 'x_cen', 'y_cen','Vsys','chi2'])
c_nfw = Table()
c_nfw['galaxy_ID'] = galaxy_ID
c_nfw['A'] = np.nan
c_nfw['Vin'] = np.nan
c_nfw['SigD'] = np.nan
c_nfw['Rd'] = np.nan
c_nfw['rho0_h'] = np.nan
c_nfw['Rh'] = np.nan
c_nfw['incl'] = np.nan
c_nfw['phi'] = np.nan
c_nfw['x_cen'] = np.nan
c_nfw['y_cen'] = np.nan
c_nfw['Vsys'] = np.nan
c_nfw['chi2'] = np.nan

# Burket
#c_bur = open('bur_exp.csv','w')
#writer_bur = csv.writer(c_bur)
#writer_bur.writerow(['galaxy_ID', 'A', 'Vin', 'SigD', 'Rd', 'rho0_h', 'Rh', 'incl', 'phi', 'x_cen', 'y_cen','Vsys','chi2'])
c_bur = Table()
c_bur['galaxy_ID'] = galaxy_ID
c_bur['A'] = np.nan
c_bur['Vin'] = np.nan
c_bur['SigD'] = np.nan
c_bur['Rd'] = np.nan
c_bur['rho0_h'] = np.nan
c_bur['Rh'] = np.nan
c_bur['incl'] = np.nan
c_bur['phi'] = np.nan
c_bur['x_cen'] = np.nan
c_bur['y_cen'] = np.nan
c_bur['Vsys'] = np.nan
c_bur['chi2'] = np.nan

c_scale = Table()
c_scale['galaxy_ID'] = galaxy_ID
c_scale['scale'] = np.nan

# Fitting the galaxy

for i in range(len(galaxy_ID)):

    plate, IFU = galaxy_ID[i].split('-')

    data_file = VEL_MAP_FOLDER + plate + '/' + IFU + '/manga-' + galaxy_ID[i] + '-MAPS-HYB10-GAU-MILESHC.fits.gz'

    j = DRP_index[galaxy_ID[i]]

    redshift = z[j]
    velocity =  redshift* c
    distance = (velocity / H_0) * 1000 #kpc
    scale = 0.5 * distance / 206265

    c_scale['scale'][i] = scale

    #incl = np.arccos(rat[j])
    cosi2 = (rat[j]**2 - q0**2)/(1 - q0**2)
    if cosi2 < 0:
        cosi2 = 0

    incl = np.arccos(np.sqrt(cosi2))

    #ph = phi[j] * np.pi / 180

    if path.exists(data_file):
        ########################################################################
        # Get data
        #-----------------------------------------------------------------------
        # scale, incl, ph, rband, Ha_vel, Ha_vel_ivar, Ha_vel_mask, vmasked, gshape, x_center_guess, y_center_guess = Galaxy_Data(galaxy_ID)
        data_maps, gshape, x_center_guess, y_center_guess = Galaxy_Data(galaxy_ID[i], 
                                                                        MANGA_FOLDER)
        #-----------------------------------------------------------------------


        ########################################################################
        # Selection
        #-----------------------------------------------------------------------
        # Morphological cut
        tidal = getTidal(galaxy_ID[i], MORPH_FOLDER)

        # Smoothness cut
        max_map_smoothness = 1.85

        map_smoothness = how_smooth(data_maps['Ha_vel'], data_maps['Ha_vel_mask'])

        SN_map = data_maps['Ha_flux'] * np.sqrt(data_maps['Ha_flux_ivar'])
        Ha_vel_mask = data_maps['Ha_vel_mask'] + (SN_map < 5)

        vmasked = ma.array(data_maps['Ha_vel'], mask = Ha_vel_mask)
        ivar_masked = ma.array(data_maps['Ha_vel_ivar'], mask = Ha_vel_mask)

        #plt.imshow(vmasked,origin='lower',cmap='RdBu_r')
        #plt.show()

        if map_smoothness <= max_map_smoothness and tidal == 0:
            
            print('Fitting galaxy ', galaxy_ID[i])
            start_time = time.time()

            center_coord = (x_center_guess, y_center_guess)

            ####################################################################
            # Find initial guess for phi
            #-------------------------------------------------------------------
            phi_guess = find_phi(center_coord, phi[j], vmasked)

            if galaxy_ID[i] in ['8134-6102']:
                phi_guess += 0.25 * np.pi

            elif galaxy_ID[i] in ['8932-12704', '8252-6103']:
                phi_guess -= 0.25 * np.pi

            elif galaxy_ID[i] in ['8613-12703', '8726-1901', '8615-1901', '8325-9102',
                                 '8274-6101', '9027-12705', '9868-12702', '8135-1901',
                                 '7815-1901', '8568-1901', '8989-1902', '8458-3701',
                                 '9000-1901', '9037-3701', '8456-6101']:
                phi_guess += 0.5 * np.pi

            elif galaxy_ID[i] in ['9864-3702', '8601-1902']:
                phi_guess -= 0.5 * np.pi

            elif galaxy_ID[i] in ['9502-12702']:
                phi_guess += 0.75 * np.pi

            elif galaxy_ID[i] in ['7495-6104']:
                phi_guess -= 0.8 * np.pi

            elif galaxy_ID[i] in ['7495-12704','7815-6103','9029-12705', '8137-3701', '8618-3704', '8323-12701',
                                 '8942-3703', '8333-12701', '8615-6103', '9486-3704',
                                 '8937-1902', '9095-3704', '8466-1902', '9508-3702',
                                 '8727-3703', '8341-12704', '8655-6103']:
                phi_guess += np.pi

            elif galaxy_ID[i] in ['7815-9102']:
                phi_guess -= np.pi

            elif galaxy_ID[i] in ['7443-9101', '7443-3704']:
                phi_guess -= 1.06 * np.pi

            elif galaxy_ID[i] in ['8082-1901', '8078-3703', '8551-1902', '9039-3703',
                                 '8624-1902', '8948-12702', '8443-6102', '8259-1901']:
                phi_guess += 1.5 * np.pi

            elif galaxy_ID[i] in ['8241-12705', '8326-6102']:
                phi_guess += 1.75 * np.pi

            elif galaxy_ID[i] in ['7443-6103']:
                phi_guess += 2.3 * np.pi

            # elif gal_ID in ['8655-1902', '7960-3701', '9864-9101', '8588-3703']:
            #     phi_guess = phi_EofN_deg * np.pi / 180.

            print(phi_guess * 180 / (np.pi))

            phi_guess = phi_guess % (2 * np.pi)
            ####################################################################


            ####################################################################
            # Fit the galaxy (normal likelihood)
            #-------------------------------------------------------------------
            parameters = [incl, phi_guess, x_center_guess, y_center_guess]

            Isothermal_fit = Galaxy_Fitting_iso(parameters,
                                                scale,
                                                gshape,
                                                vmasked,
                                                ivar_masked,
                                                Ha_vel_mask)
            
            NFW_fit = Galaxy_Fitting_NFW(parameters,
                                         scale,
                                         gshape,
                                         vmasked,
                                         ivar_masked,
                                         Ha_vel_mask)

            Burket_fit = Galaxy_Fitting_bur(parameters,
                                            scale,
                                            gshape,
                                            vmasked,
                                            ivar_masked,
                                            Ha_vel_mask)
            
            print('Fit galaxy', time.time() - start_time)
            ####################################################################



            ####################################################################
            # Plotting
            #-------------------------------------------------------------------
            #Plotting_Isothermal(galaxy_ID[i], gshape, scale, Isothermal_fit, Ha_vel_mask)
            #Plotting_NFW(galaxy_ID[i], gshape, scale, NFW_fit, Ha_vel_mask)
            #Plotting_Burket(galaxy_ID[i], gshape, scale, Burket_fit, Ha_vel_mask)

            #plot_fits(galaxy_ID[i], vmasked, ivar_masked, scale, gshape, Isothermal_fit, NFW_fit, Burket_fit, Ha_vel_mask)
            
            #plot_rot_curve(vmasked,ivar_masked,Isothermal_fit,scale,galaxy_ID[i],'Isothermal')
            #plot_rot_curve(vmasked,ivar_masked,NFW_fit,scale,galaxy_ID[i],'NFW')
            #plot_rot_curve(vmasked,ivar_masked,Burket_fit,scale,galaxy_ID[i],'Burket')
            
            Isothermal_fit = np.ndarray.tolist(Isothermal_fit)
            NFW_fit = np.ndarray.tolist(NFW_fit)
            Burket_fit = np.ndarray.tolist(Burket_fit)
            
            plot_diagnostic_panel(galaxy_ID[i], gshape, scale, Isothermal_fit, NFW_fit, Burket_fit, data_maps['Ha_vel_mask'], data_maps['vmasked'], data_maps['ivar_masked'])
            ####################################################################




            ####################################################################
            # Calculating Chi2
            #-------------------------------------------------------------------
            print('Calculating chi2')
            start_time = time.time()

            #-------------------------------------------------------------------
            # Isothermal
            full_vmap_iso = rot_incl_iso(gshape, scale, Isothermal_fit)

            # Masked array
            vmap_iso = ma.array(full_vmap_iso, mask=Ha_vel_mask)

            # need to calculate number of data points in the fitted map
            # nd_iso = vmap_iso.shape[0]*vmap_iso.shape[1] - np.sum(vmap_iso.mask)
            nd_iso = np.sum(~vmap_iso.mask)

            # chi2_iso = np.nansum((vmasked - vmap_iso) ** 2 * Ha_vel_ivar)
            chi2_iso = ma.sum(data_maps['Ha_vel_ivar'] * (vmasked - vmap_iso) ** 2)

            # chi2_iso_norm = chi2_iso/(nd_iso - 8)
            chi2_iso_norm = chi2_iso / (nd_iso - len(Isothermal_fit))
            #-------------------------------------------------------------------

            
            #-------------------------------------------------------------------
            # NFW
            full_vmap_NFW = rot_incl_NFW(gshape, scale, NFW_fit)

            # Masked array
            vmap_NFW = ma.array(full_vmap_NFW, mask=Ha_vel_mask)

            # need to calculate number of data points in the fitted map
            # nd_NFW = vmap_NFW.shape[0]*vmap_NFW.shape[1] - np.sum(vmap_NFW.mask)
            nd_NFW = np.sum(~vmap_NFW.mask)

            # chi2_NFW = np.nansum((vmasked - vmap_NFW) ** 2 * Ha_vel_ivar)
            chi2_NFW = ma.sum(data_maps['Ha_vel_ivar'] * (vmasked - vmap_NFW) ** 2)

            # chi2_NFW_norm = chi2_NFW/(nd_NFW - 8)
            chi2_NFW_norm = chi2_NFW / (nd_NFW - len(NFW_fit))
            #-------------------------------------------------------------------


            #-------------------------------------------------------------------
            # Burket
            full_vmap_bur = rot_incl_bur(gshape, scale, Burket_fit)

            # Masked array
            vmap_bur = ma.array(full_vmap_bur, mask=Ha_vel_mask)

            # need to calculate number of data points in the fitted map
            # nd_bur = vmap_bur.shape[0]*vmap_bur.shape[1] - np.sum(vmap_bur.mask)
            nd_bur = np.sum(~vmap_bur.mask)

            # chi2_bur = np.nansum((vmasked - vmap_bur) ** 2 * Ha_vel_ivar)
            chi2_bur = ma.sum(data_maps['Ha_vel_ivar'] * (vmasked - vmap_bur) ** 2)

            # chi2_bur_norm = chi2_bur/(nd_bur-8)
            chi2_bur_norm = chi2_bur / (nd_bur - len(Burket_fit))
            #-------------------------------------------------------------------
            ####################################################################


            ####################################################################
            # Write results to file
            #-------------------------------------------------------------------
            '''
            writer_iso.writerow([galaxy_ID[i], 
                                 Isothermal_fit[0], 
                                 Isothermal_fit[1], 
                                 Isothermal_fit[2], 
                                 Isothermal_fit[3], 
                                 Isothermal_fit[4], 
                                 Isothermal_fit[5], 
                                 Isothermal_fit[6], 
                                 Isothermal_fit[7], 
                                 Isothermal_fit[8], 
                                 Isothermal_fit[9], 
                                 Isothermal_fit[10], 
                                 chi2_iso_norm])
            '''
            c_iso['A'][i] = Isothermal_fit[0]
            c_iso['Vin'][i] = Isothermal_fit[1]
            c_iso['SigD'][i] = Isothermal_fit[2]
            c_iso['Rd'][i] = Isothermal_fit[3]
            c_iso['rho0_h'][i] = Isothermal_fit[4]
            c_iso['Rh'][i] = Isothermal_fit[5]
            c_iso['incl'][i] = Isothermal_fit[6]
            c_iso['phi'][i] = Isothermal_fit[7]
            c_iso['x_cen'][i] = Isothermal_fit[8]
            c_iso['y_cen'][i] = Isothermal_fit[9]
            c_iso['Vsys'][i] = Isothermal_fit[10]
            c_iso['chi2'][i] = chi2_iso_norm
            
            '''
            writer_nfw.writerow([galaxy_ID[i], 
                                 NFW_fit[0], 
                                 NFW_fit[1], 
                                 NFW_fit[2], 
                                 NFW_fit[3], 
                                 NFW_fit[4], 
                                 NFW_fit[5], 
                                 NFW_fit[6], 
                                 NFW_fit[7], 
                                 NFW_fit[8], 
                                 NFW_fit[9], 
                                 NFW_fit[10], 
                                 chi2_NFW_norm])
            '''
            c_nfw['A'][i] = NFW_fit[0]
            c_nfw['Vin'][i] = NFW_fit[1]
            c_nfw['SigD'][i] = NFW_fit[2]
            c_nfw['Rd'][i] = NFW_fit[3]
            c_nfw['rho0_h'][i] = NFW_fit[4]
            c_nfw['Rh'][i] = NFW_fit[5]
            c_nfw['incl'][i] = NFW_fit[6]
            c_nfw['phi'][i] = NFW_fit[7]
            c_nfw['x_cen'][i] = NFW_fit[8]
            c_nfw['y_cen'][i] = NFW_fit[9]
            c_nfw['Vsys'][i] = NFW_fit[10]
            c_nfw['chi2'][i] = chi2_NFW_norm

            '''
            writer_bur.writerow([galaxy_ID[i], 
                                 Burket_fit[0],
                                 Burket_fit[1],
                                 Burket_fit[2],
                                 Burket_fit[3],
                                 Burket_fit[4],
                                 Burket_fit[5],
                                 Burket_fit[6],
                                 Burket_fit[7],
                                 Burket_fit[8],
                                 Burket_fit[9],
                                 Burket_fit[10],
                                 chi2_bur_norm])
            '''
            c_bur['A'][i] = Burket_fit[0]
            c_bur['Vin'][i] = Burket_fit[1]
            c_bur['SigD'][i] = Burket_fit[2]
            c_bur['Rd'][i] = Burket_fit[3]
            c_bur['rho0_h'][i] = Burket_fit[4]
            c_bur['Rh'][i] = Burket_fit[5]
            c_bur['incl'][i] = Burket_fit[6]
            c_bur['phi'][i] = Burket_fit[7]
            c_bur['x_cen'][i] = Burket_fit[8]
            c_bur['y_cen'][i] = Burket_fit[9]
            c_bur['Vsys'][i] = Burket_fit[10]
            c_bur['chi2'][i] = chi2_bur_norm
            ####################################################################

            ####################################################################
            print('Isothermal chi2:', chi2_iso_norm, time.time() - start_time)
            print('NFW chi2:', chi2_NFW_norm)
            print('Burket chi2:', chi2_bur_norm)
            ####################################################################

            '''
            ####################################################################
            # Calculating the Hessian Matrix
            #-------------------------------------------------------------------
            print('Calculating Hessian')
            start_time = time.time()

            Hessian_iso = Hessian_Calculation_Isothermal(Isothermal_fit,
                                                         scale,
                                                         gshape,
                                                         vmasked,
                                                         Ha_vel_ivar)

            Hessian_NFW = Hessian_Calculation_NFW(NFW_fit,
                                                  scale,
                                                  gshape,
                                                  vmasked,
                                                  Ha_vel_ivar)

            Hessian_bur = Hessian_Calculation_Burket(Burket_fit,
                                                     scale,
                                                     gshape,
                                                     vmasked,
                                                     Ha_vel_ivar)

            print('Calculated Hessian', time.time() - start_time)
            ####################################################################
            '''
        
        else:
            print(galaxy_ID[i] + ' does not have rotation curve')
            '''
            writer_iso.writerow([galaxy_ID[i], 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A',
                                 'N/A',
                                 'N/A'])
            writer_nfw.writerow([galaxy_ID[i], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A','N/A','N/A'])
            writer_bur.writerow([galaxy_ID[i], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A','N/A','N/A'])
            '''
    else:
        print('No data for the galaxy.')
        '''
        writer_iso.writerow([galaxy_ID[i], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A','N/A'])
        writer_nfw.writerow([galaxy_ID[i], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A','N/A'])
        writer_bur.writerow([galaxy_ID[i], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A','N/A'])
        '''
'''
c_iso.close()
c_nfw.close()
c_bur.close()
'''

c_iso.write('iso_exp.csv', format='ascii.csv', overwrite=True)
c_nfw.write('nfw_exp.csv', format='ascii.csv', overwrite=True)
c_bur.write('bur_exp.csv', format='ascii.csv', overwrite=True)
c_scale.write('gal_scale.csv', format='ascii.csv',overwrite=True)
