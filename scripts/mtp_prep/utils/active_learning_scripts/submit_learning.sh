#!/bin/bash
##SBATCH --exclusive
#SBATCH --array=1-88 ## change accordingly
#SBATCH --mail-type=None
#SBATCH --mem-per-cpu=3800
#SBATCH -A project01915 
#SBATCH -C avx512
#SBATCH -J MTP_active
#SBATCH -e err/lmp_%A_%a.err
#SBATCH -n 1
#SBATCH -o out/lmp_%A_%a.out
#SBATCH -t 24:00:00

#---------------
module purge
module use /home/groups/da_mm/modules
export MM_APPS_ROOT=/home/groups/da_mm/apps
export MM_MODULES_ROOT=/home/groups/da_mm/modules

###########################################
# LAMMPS with active learning
# !!! Make sure that mlip.ini and mlip.active.ini point to the right potential and state.als files !!!

module load lammps_ml
unset I_MPI_PMI_LIBRARY
echo $SLURM_ARRAY_TASK_ID

\cp mlip.active.ini cook_and_quench.in $SLURM_ARRAY_TASK_ID
cd $SLURM_ARRAY_TASK_ID

srun lmp -in  cook_and_quench.in