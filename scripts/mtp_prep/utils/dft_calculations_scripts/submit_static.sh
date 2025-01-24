#!/bin/bash
#SBATCH --mail-type=None
#SBATCH --array=401-930
#SBATCH --mem-per-cpu=3800
#SBATCH -A project01915   # CHANGE
#SBATCH -C avx512
#SBATCH -J static_calc_pbe_bto
#SBATCH -e err/%A_%a.err
#SBATCH -o out/%A_%a.out
#SBATCH -n 24          # CHANGE
#SBATCH -t 24:00:00


#---------------


module purge
module use /home/groups/da_mm/modules
export MM_APPS_ROOT=/home/groups/da_mm/apps
export MM_MODULES_ROOT=/home/groups/da_mm/modules
module load vasp_vtst
export VASP_SCRIPT=/home/as41vomu/vasp/run_vasp.py
export VASP_PP_PATH=/home/as41vomu/vasp/mypps

#--------------------
cp vasp_scripts/run.py $SLURM_ARRAY_TASK_ID
cd $SLURM_ARRAY_TASK_ID

~/miniconda3/bin/python run.py

exit_code=$?
echo ""
echo "Executable 'vasp' finished with exit code $exit_code"
