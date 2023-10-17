import os
import numpy as np
import numpy.ma as ma
from astropy.table import Table
from scipy.optimize import minimize, Bounds
from DRP_rotation_curve import extract_data, extract_Pipe3d_data
from disk_mass import calc_mass_curve, fit_mass_curve
from rotation_fitfunctions import find_phi, parameterfit_iso, parameterfit_NFW, parameterfit_bur

FILE_IDS = ['7443-12705']
H_0 = 100  # Hubble's Constant in units of h km/s/Mpc
c = 299792.458  # Speed of light in units of km/s
fit_function = 'bulge'
LOCAL_PATH = os.path.dirname(__file__)
MANGA_FOLDER = r"C:\Users\Lara\Documents\rotationcurves\mangadata"
NSA_FILENAME = r"C:\Users\Lara\Documents\rotationcurves\nsa_v1_0_1.fits"

MASS_MAP_FOLDER = r"C:\Users\Lara\Documents\rotationcurves\mangadata\pipe3d\v3_1_1\3.1.1\7443"
VEL_MAP_FOLDER = r"C:\Users\Lara\Documents\rotationcurves\mangadata\analysis\v3_1_1\3.1.0\HYB10-MILESHC-MASTARSSP\7443\12705"
DRP_FILENAME = r"C:\Users\Lara\Documents\rotationcurves\mangadata\redux\v3_1_1\drpall-v3_1_1.fits"
DRP_table = Table.read(DRP_FILENAME, format='fits')

DRP_index = {}

for i in range(len(DRP_table)):
    gal_ID = DRP_table['plateifu'][i]

    DRP_index[gal_ID] = i
for gal_ID in FILE_IDS:
    best_vfit_set = []
    best_diskfit_set = []
    maps = extract_data(VEL_MAP_FOLDER,
                        gal_ID,
                        ['Ha_vel', 'r_band', 'Ha_flux', 'Ha_sigma'])
    sMass_density, sMass_density_err = extract_Pipe3d_data(MASS_MAP_FOLDER, gal_ID)
    i_DRP = DRP_index[gal_ID]
    axis_ratio = DRP_table['nsa_elpetro_ba'][i_DRP]
    incl = np.arccos(axis_ratio)
    phi = DRP_table['nsa_elpetro_phi'][i_DRP]
    z = DRP_table['nsa_z'][i_DRP]
    map_mask = maps['Ha_vel_mask']

    maps['vmasked'] = ma.array(maps['Ha_vel'], mask=map_mask)
    maps['ivarmasked'] = ma.array(maps['Ha_vel_ivar'], mask=map_mask)

    shape = maps['vmasked'].shape
    scale = (0.5 * z * c / H_0) * 1000 / 206265  # kpc
    center = np.unravel_index(ma.argmax(maps['r_band']), shape)
    x_center = center[0]
    y_center = center[1]
    count = 0
    phi = find_phi(center, phi, maps['vmasked'])
param = [incl, phi, x_center, y_center]

best_fit_bur = parameterfit_bur(param, param_outputs[195061856.85765797], param_outputs[2.323128917886229],
                                        param_outputs[877.5968281404483], param_outputs[0.7689406307205499], scale, shape,
                                        maps['vmasked'], maps['ivarmasked'], map_mask)