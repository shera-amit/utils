#!/bin/bash
##SBATCH --exclusive
#SBATCH --mail-type=None
#SBATCH --mem-per-cpu=3800
#SBATCH --account=p0020994.
#SBATCH -C avx512
#SBATCH -J DFT_calc
#SBATCH -e err.%J
#SBATCH -n 96
#SBATCH -o out.%J
#SBATCH -t 24:00:00

module purge
module use /home/groups/da_mm/modules
export MM_APPS_ROOT=/home/groups/da_mm/apps
export MM_MODULES_ROOT=/home/groups/da_mm/modules
module load vasp_vtst
srun /work/groups/da_mm/apps/vasp_vtst/6.3.0/bin/vasp_std
