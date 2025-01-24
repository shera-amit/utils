#!/bin/bash
##SBATCH --exclusive
#SBATCH --mail-type=None
#SBATCH --mem-per-cpu=3800
#SBATCH -A project01915
#SBATCH -C avx512
#SBATCH -J MTP_fit
#SBATCH -e err.%j
#SBATCH -n 96
#SBATCH -o out.%j
#SBATCH -t 24:00:00

#---------------
module purge
module use /home/groups/da_mm/modules
export MM_APPS_ROOT=/home/groups/da_mm/apps
export MM_MODULES_ROOT=/home/groups/da_mm/modules

module load mlip
srun mlp train 18.mtp dft_0.cfg --trained-pot-name=BTO_0.mtp