#!/bin/bash
#SBATCH -A p0020994
#SBATCH -J PACEMAKER
#SBATCH --mail-type=NONE
#SBATCH -e err.%j
#SBATCH -o out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --gres=gpu
#SBATCH --mem-per-cpu=3800
#SBATCH -t 24:00:00
#SBATCH --no-requeue

# -------------------------------
module purge
module load pacemaker
#source /home/as41vomu/.bashrc
#conda activate ace
#module load cuda cuDNN intel
start=`date +%s`
srun pacemaker --verbose-tf input.yaml
end=`date +%s`
runtime=$((end-start))
echo "$runtime"

