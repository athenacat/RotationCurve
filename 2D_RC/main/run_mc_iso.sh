#!/bin/bash                                                                                                              
#SBATCH -p standard                                                                                                      
#SBATCH --mem=40gb                                                                                                       
#SBATCH --time=120:00:00                                                                                                  
#SBATCH -o /scratch/yzh250/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/manga_mc_iso_log.log
\
                                                                                                                         
#SBATCH -e /scratch/yzh250/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/manga_mc_iso_err.err
\
                                                                                                                         
                                                                                                                        \

module load anaconda3/2020.11

srun python vel_map_RC_MCMC_iso.py
