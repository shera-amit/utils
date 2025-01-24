import os
from ase.calculators.vasp import Vasp
import subprocess

class VaspRun:
    def __init__(self, calc: Vasp, atoms, job_name='DFT_calc', account='p0020994', directory=None):
        self.calc = calc
        self.atoms = atoms
        self.job_name = job_name
        self.account = account
        self.default_cores = 96
        self.default_time = 24.0
        # Check if calc has directory set, otherwise use the provided directory
        if directory is not None:
            self.directory = os.path.abspath(directory)
        else:
            self.directory = os.path.abspath(calc.directory) if calc.directory is not None else os.path.abspath('.')

    def write_sbatch_file(self, cores, time):
        hours = int(time)
        minutes = int((time - hours) * 60)
        sbatch_content = f"""#!/bin/bash
#SBATCH --exclusive
#SBATCH --mail-type=None
#SBATCH --mem-per-cpu=3800
#SBATCH --account={self.account}
#SBATCH -C avx512
#SBATCH -J {self.job_name}
#SBATCH -e err.%J
#SBATCH -n {cores}
#SBATCH -o out.%J
#SBATCH -t {hours:02}:{minutes:02}:00
module purge
module use /home/groups/da_mm/modules
export MM_APPS_ROOT=/home/groups/da_mm/apps
export MM_MODULES_ROOT=/home/groups/da_mm/modules
module load vasp_vtst
srun /work/groups/da_mm/apps/vasp_vtst/6.3.0/bin/vasp_std
"""
        with open(os.path.join(self.directory, 'submit.sh'), 'w') as f:
            f.write(sbatch_content)

    def run(self, time=None, cores=None, rerun=False):
        if time is None:
            time = self.default_time
        if cores is None:
            cores = self.default_cores

        outcar_path = os.path.join(self.directory, 'OUTCAR')
        
        if os.path.exists(outcar_path) and not rerun:
            print(f"OUTCAR file already exists in {self.directory}. Calculation not submitted.")
            print("To rerun the calculation, use rerun=True")
            return

        # Attach the calculator to atoms and set directory, then write necessary input files
        self.calc.directory = self.directory
        self.atoms.set_calculator(self.calc)
        self.calc.write_input(self.atoms)

        # Write sbatch file
        self.write_sbatch_file(cores, time)

        # Print the absolute path where the job will be run
        print(f"Submitting job from directory: {os.path.abspath(self.directory)}")

        # Submit job to HPC
        try:
            subprocess.run(['sbatch', 'submit.sh'], check=True, cwd=self.directory)
            print("Job submitted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error in job submission: {e}")

if __name__ == "__main__":
    # Usage example:
    # Define your VASP calculator
    calc = Vasp(xc='PBE', encut=520, kpts=[3, 3, 3])
    # Define your atoms object (example, you should define your atoms)
    # atoms = ...
    # Create VaspRun instance
    # vasp_run = VaspRun(calc, atoms, directory='my_calc_directory')
    # Run the calculation
    # vasp_run.run(time=0.5, cores=24)  # Set custom time (in hours) and cores (number of cores) as needed
    # To rerun an existing calculation:
    # vasp_run.run(time=0.5, cores=24, rerun=True)
    pass
