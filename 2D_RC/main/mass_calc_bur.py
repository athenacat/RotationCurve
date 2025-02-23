################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np
from astropy.table import Table, QTable
from astropy.io import ascii

from mass_calc_functions import bulge_mass,\
								disk_mass,\
								halo_mass_bur
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
# Used files (local)
#-------------------------------------------------------------------------------
MANGA_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'

DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/drpall-v3_1_1.fits'

# Can't really use this anymore
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis//v3_1_1/2.1.1/HYB10-GAU-MILESHC/'

MORPH_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/morph/'

fits_file = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/Notebooks/'

main_file = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/'

r90_file = main_file + 'r90.csv'

logHI_file = fits_file + 'logHI.npy'

logH2_file = fits_file + 'logH2.npy'
################################################################################

################################################################################
# Used files (bluehive)
#-------------------------------------------------------------------------------
'''
MANGA_FOLDER_yifan = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/spectro/'

DRP_FILENAME = MANGA_FOLDER_yifan + 'redux/v3_1_1/drpall-v3_1_1.fits'

VEL_MAP_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'

MORPH_FOLDER = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/morph/'

fits_file = '/home/yzh250/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/'
'''
################################################################################

################################################################################
# DRP all table
DTable =  Table.read(DRP_FILENAME, format='fits')

z = DTable['nsa_z']

f_r90 = ascii.read(r90_file,'r')

r90_list = f_r90['r90']

DRP_index = {}

for i in range(len(DTable)):
    gal_ID = DTable['plateifu'][i]

    DRP_index[gal_ID] = i

galaxy_ID = []
plateifu = DTable['plateifu'].data

for i in range(len(plateifu)):
    galaxy_ID.append(str(plateifu[i],'utf-8'))

stellar_mass = DTable['nsa_elpetro_mass']

logHI = np.load(logHI_file)

logH2 = np.load(logH2_file)
################################################################################

################################################################################
# Reading in all the files
fit_mini_bur_name = fits_file + 'bur_mini.csv'
fit_mini_bur = ascii.read(fit_mini_bur_name,'r')
################################################################################

c_mass_bur = Table()
c_mass_bur['galaxy_ID'] = galaxy_ID
c_mass_bur['Mb'] = np.nan
c_mass_bur['Md'] = np.nan
c_mass_bur['Mh_r90'] = np.nan
c_mass_bur['Mh_2r90'] = np.nan
c_mass_bur['Mh_3r90'] = np.nan
c_mass_bur['M_HI'] = np.nan
c_mass_bur['M_HII'] = np.nan
c_mass_bur['Mtot_r90'] = np.nan
c_mass_bur['Mtot_2r90'] = np.nan
c_mass_bur['Mtot_3r90'] = np.nan
c_mass_bur['NSA_Mstar'] = stellar_mass
c_mass_bur['Mtot_(star_HI_Hii)'] = np.nan
c_mass_bur['M_h1'] = np.nan
c_mass_bur['M_h2'] = np.nan
c_mass_bur['Mtot_(star_HI_Hii)'] = np.nan

################################################################################
for i in range(len(fit_mini_bur)):
    gal_fit = list(fit_mini_bur[i])

    j = DRP_index[gal_fit[0]]
    redshift = z[j]
    velocity = redshift * c
    distance = (velocity / H_0) * 1000
    r90 = distance * (r90_list[i]/206265)
    Burket_fit_mini = gal_fit[1:-2]
    #print(Isothermal_fit_mini)

    MHI = 10**logHI[i]
    MH2 = 10**logH2[i]

    if np.isfinite(gal_fit[-1]) and gal_fit[-1] < 150:
        rho_0 = float(Burket_fit_mini[0])
        Rb = float(Burket_fit_mini[1])
        SigD = float(Burket_fit_mini[2])
        Rd = float(Burket_fit_mini[3])
        rho0_h = float(Burket_fit_mini[4])
        Rh = float(Burket_fit_mini[5])
        if round(rho_0) == -7 or round(rho_0) == 1\
           or round(Rb) == 0 or round(Rb) == 10\
           or round(SigD,1) == 0.1 or round(SigD,1) == 3000\
           or round(Rd,1) == 0.1 or round(Rd) == 30\
           or round(rho0_h) == -7 or round(rho0_h) == 2\
           or round(Rh,1) == 0.1 or round(Rh) == 1000:
           pass
        else:
            Mb = bulge_mass(rho_0,Rb)
            Md = disk_mass(SigD,Rd)
            Mh_r90 = halo_mass_bur(rho0_h,r90,Rh)
            Mh_2r90 = halo_mass_bur(rho0_h,2*r90,Rh)
            Mh_3r90 = halo_mass_bur(rho0_h,3*r90,Rh)
            Mtot_r90 = Mb + Md + Mh_r90
            Mtot_2r90 = Mb + Md + Mh_2r90
            Mtot_3r90 = Mb + Md + Mh_3r90
            Mtot2 = stellar_mass[i] + MHI + MH2 
            c_mass_bur['Mb'][i] = Mb
            c_mass_bur['Md'][i] = Md
            c_mass_bur['Mh_r90'][i] = Mh_r90
            c_mass_bur['Mh_2r90'][i] = Mh_2r90
            c_mass_bur['Mh_3r90'][i] = Mh_3r90
            c_mass_bur['Mtot_r90'][i] = Mtot_r90
            c_mass_bur['Mtot_2r90'][i] = Mtot_2r90
            c_mass_bur['Mtot_3r90'][i] = Mtot_3r90
            c_mass_bur['M_h1'][i] = MHI
            c_mass_bur['M_h2'][i] = MH2
            c_mass_bur['Mtot_(star_HI_Hii)'][i] = Mtot2
c_mass_bur.write('mass_bur.csv', format='ascii.csv', overwrite=True)