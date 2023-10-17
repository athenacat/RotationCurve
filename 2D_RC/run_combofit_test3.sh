#!/bin/bash          


#SBATCH -p standard                                                                                                      
#SBATCH --mem=40gb                                                                                                       
#SBATCH --time=120:00:00  




#SBATCH -o /scratch/lstroud3/RotationCurves/combofitIsoFixfindphi.log
                                                                                                                         
#SBATCH -e /scratch/lstroud3/RotationCurves/combofitIsoFixfindphi.err
#SBATCH --mail-user=lstroud3@u.rochester.edu                                                                                                                       

#SBATCH --mail-type=ALL                                                                                                                         
                                                                                                                        \
module unload python3
module load anaconda3/2021.11

srun python combo_fit_fixed_centers.py Burkert
