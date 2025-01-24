#!/bin/bash
##SBATCH --exclusive
#SBATCH --mail-type=None
#SBATCH --mem-per-cpu=3800
#SBATCH -A project01915 
#SBATCH -C avx512
#SBATCH -J MTP_active
#SBATCH -e err.%J 
#SBATCH -n 24
#SBATCH -o out.%J
#SBATCH -t 24:00:00

#---------------


module purge
module use /home/groups/da_mm/modules
export MM_APPS_ROOT=/home/groups/da_mm/apps
export MM_MODULES_ROOT=/home/groups/da_mm/modules
module load vasp_vtst
export VASP_SCRIPT=/home/as41vomu/vasp/run_vasp.py
export VASP_PP_PATH=/home/as41vomu/vasp/mypps


echo $VASP_SCRIPT
echo $VASP_PP_PATH
echo $HOME
#--------------------
~/miniconda3/bin/python run_relax.py
exit_code=$?
echo ""
echo "Executable 'vasp' finished with exit code $exit_code"

