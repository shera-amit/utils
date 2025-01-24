#!/bin/bash
##SBATCH --exclusive
#SBATCH --array=1-88
#SBATCH --mail-type=None
#SBATCH --mem-per-cpu=3800
#SBATCH -A project01915 
#SBATCH -C avx512
#SBATCH -J MTP_active
#SBATCH -e err/red_ind_%A_%a.err
#SBATCH -n 1
#SBATCH -o out/red_ind_%A_%a.out
#SBATCH -t 04:00:00

#---------------
module purge
module use /home/groups/da_mm/modules
export MM_APPS_ROOT=/home/groups/da_mm/apps
export MM_MODULES_ROOT=/home/groups/da_mm/modules

##########################################
## reduce number of training structures in individual subdirs
## !!! Make sure that you specify the right potential / training DB !!!

module load mlip
unset I_MPI_PMI_LIBRARY

cd $SLURM_ARRAY_TASK_ID
srun mlp select-add  ../../mtp_fit/BTO_0.mtp ../../mtp_fit/dft_0.cfg  B-preselected.cfg C-add_to_train.cfg 