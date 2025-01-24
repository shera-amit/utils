#!/bin/bash
##SBATCH --exclusive
#SBATCH --mail-type=None
#SBATCH --mem-per-cpu=3800
#SBATCH -A project01915 
#SBATCH -C avx512
#SBATCH -J MTP_fit
#SBATCH -e err.%j
#SBATCH -n 1
#SBATCH -o out.%j
#SBATCH -t 00:30:00

#---------------
module purge
module use /home/groups/da_mm/modules
export MM_APPS_ROOT=/home/groups/da_mm/apps
export MM_MODULES_ROOT=/home/groups/da_mm/modules

module load mlip

# Create .als file for active learning
# use one core only!!!
unset I_MPI_PMI_LIBRARY 
srun mlp calc-grade BTO_0.mtp dft_0.cfg dft_0.cfg temp.cfg --als-filename=state.als