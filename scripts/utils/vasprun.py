import os
import logging
from ase.calculators.vasp import Vasp
import subprocess
from datetime import datetime

# Configure logging at the module level
logger = logging.getLogger('VaspRun')
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    log_file = os.path.join(os.getcwd(), 'vasp_run.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

class VaspRun:
    def __init__(self, calc: Vasp, atoms, job_name='DFT_calc', account='p0020994', directory=None):
        self.calc = calc
        self.atoms = atoms
        self.job_name = job_name
        self.account = account
        self.default_cores = 96
        self.default_time = 24.0
        self.logger = logger

        # Set up the base directory and working directory
        if directory is not None:
            self.base_directory = os.path.abspath(directory)
        else:
            self.base_directory = os.path.abspath('.')
            
        self.directory = os.path.join(self.base_directory, self.job_name)
        os.makedirs(self.directory, exist_ok=True)

        # Log initial setup
        self.logger.info(f"{'='*50}")
        self.logger.info(f"VaspRun initialized for job: {job_name}")
        self.logger.info(f"Working directory: {self.directory}")
        self.logger.info(f"{'='*50}")

    def write_files(self):
        """Write VASP input files only."""
        try:
            outcar_path = os.path.join(self.directory, 'OUTCAR')
            if os.path.exists(outcar_path):
                self.logger.warning(f"OUTCAR already exists in {self.directory}")
                if self.check_convergence(outcar_path):
                    self.logger.warning("Previous calculation has converged")
                    return False
                self.logger.info("Previous calculation not converged, proceeding with new input files")
            
            self.calc.directory = self.directory
            self.atoms.set_calculator(self.calc)
            self.calc.write_input(self.atoms)
            self.logger.info("VASP input files written successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing input files: {e}")
            return False

    def write_sbatch_file(self, cores, time):
        """Write SLURM batch script."""
        try:
            hours = int(time)
            minutes = int((time - hours) * 60)
            
            sbatch_content = f"""#!/bin/bash
#SBATCH --exclusive
#SBATCH --mail-type=None
#SBATCH --mem-per-cpu=3800
#SBATCH --account={self.account}
#SBATCH -C avx512
#SBATCH -J {self.job_name}
#SBATCH -e {self.job_name}.err.%J
#SBATCH -n {cores}
#SBATCH -o {self.job_name}.out.%J
#SBATCH -t {hours:02}:{minutes:02}:00

rm -rf $HOME/.cache/lmod/* ; chmod 0500 $HOME/.cache/lmod
module purge
module use /home/groups/da_mm/modules
export MM_APPS_ROOT=/home/groups/da_mm/apps
export MM_MODULES_ROOT=/home/groups/da_mm/modules
module load vasp_vtst
srun /work/groups/da_mm/apps/vasp_vtst/6.3.0/bin/vasp_std"""

            sbatch_path = os.path.join(self.directory, 'submit.sh')
            with open(sbatch_path, 'w') as f:
                f.write(sbatch_content)
            self.logger.info(f"SBATCH file written to {sbatch_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error writing sbatch file: {e}")
            return False

    def check_existing_jobs(self):
        """Check for existing jobs in the directory."""
        user = os.environ.get('USER')
        if not user:
            self.logger.error("Environment variable USER is not set")
            return []
            
        try:
            result = subprocess.run(['squeue', '-u', user, '--format=%i %Z'], 
                                 capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            jobs_in_directory = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    jobid = parts[0]
                    job_dir = parts[1]
                    if os.path.abspath(job_dir) == self.directory:
                        jobs_in_directory.append(jobid)
                        
            if jobs_in_directory:
                self.logger.info(f"Found existing jobs: {', '.join(jobs_in_directory)}")
            else:
                self.logger.info("No existing jobs found")
            return jobs_in_directory
            
        except Exception as e:
            self.logger.error(f"Error checking existing jobs: {str(e)}")
            return []

    def check_convergence(self, outcar_path):
        """Check VASP calculation convergence status."""
        try:
            if not os.path.exists(outcar_path):
                return False
            with open(outcar_path, 'r') as f:
                lines = f.readlines()[-100:]
            converged = 'reached required accuracy' in ''.join(lines)
            self.logger.info(f"Convergence check: {'Converged' if converged else 'Not converged'}")
            return converged
        except Exception as e:
            self.logger.error(f"Error checking convergence: {str(e)}")
            return False

    def run(self, time=None, cores=None, rerun=False):
        """Execute VASP calculation with specified parameters."""
        time = time or self.default_time
        cores = cores or self.default_cores
        
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Starting new VASP run")
        self.logger.info(f"Parameters: time={time}h, cores={cores}, rerun={rerun}")

        # Check existing calculation
        outcar_path = os.path.join(self.directory, 'OUTCAR')
        if os.path.exists(outcar_path):
            converged = self.check_convergence(outcar_path)
            if converged and not rerun:
                msg = "Calculation already converged. Use rerun=True to force restart."
                self.logger.warning(msg)
                print(f"WARNING: {msg}")
                return False

        # Check running jobs
        existing_jobs = self.check_existing_jobs()
        if existing_jobs and not rerun:
            msg = f"Active jobs found: {', '.join(existing_jobs)}. Use rerun=True to cancel and restart."
            self.logger.warning(msg)
            print(f"WARNING: {msg}")
            return False
        elif existing_jobs and rerun:
            for job_id in existing_jobs:
                try:
                    subprocess.run(['scancel', job_id], check=True)
                    self.logger.info(f"Cancelled job {job_id}")
                except Exception as e:
                    self.logger.error(f"Failed to cancel job {job_id}: {str(e)}")

        # Write required files
        if not self.write_files():
            return False
        if not self.write_sbatch_file(cores, time):
            return False

        # Submit job
        try:
            result = subprocess.run(['sbatch', 'submit.sh'], 
                                 check=True, cwd=self.directory, 
                                 capture_output=True, text=True)
            job_id = result.stdout.strip().split()[-1]
            self.logger.info(f"Job submitted successfully (ID: {job_id})")
            print(f"Job {job_id} submitted to {self.directory}")
            return True
        except Exception as e:
            self.logger.error(f"Job submission failed: {str(e)}")
            return False

if __name__ == "__main__":
    # Usage example:
    """
    calc = Vasp(xc='PBE', encut=520, kpts=[3, 3, 3])
    atoms = ... # Your atoms object
    
    # Initialize
    vasp_run = VaspRun(calc, atoms, job_name='test_calc', directory='calculations')
    
    # Write input files only
    vasp_run.write_files()
    
    # Or run full calculation
    vasp_run.run(time=12, cores=48)
    
    # Force rerun
    vasp_run.run(time=12, cores=48, rerun=True)
    """
    pass
