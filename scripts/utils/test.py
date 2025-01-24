import os
import logging
import subprocess
from ase.io.lammpsdata import write_lammps_data
import shutil
from datetime import datetime

class LmpRun:
    def __init__(self, lmp_script, atoms, directory=None, job_name='LAMMPS_calc',
                 pot='ace', pot_path=None, element_order=None, account='p0020994'):
        """
        Initialize LmpRun for LAMMPS calculations.

        Args:
            lmp_script (str): LAMMPS input script content or path to script file
            atoms: Atomic structure object
            directory (str, optional): Working directory. Defaults to current directory
            job_name (str, optional): Name of the job. Defaults to 'LAMMPS_calc'
            pot (str, optional): Potential type. Defaults to 'ace'
            pot_path (str, optional): Path to potential file
            element_order (list, optional): Order of elements
            account (str, optional): Account for job submission
        """
        self.atoms = atoms
        self.job_name = job_name
        self.account = account
        self.pot = pot
        self.pot_path = pot_path
        self.element_order = element_order or []
        self.default_cores = 96
        self.default_time = 24.0

        # Set up the directory
        self.directory = os.path.abspath(os.path.join(directory or '.', job_name))
        os.makedirs(self.directory, exist_ok=True)

        # Configure logging for this instance
        self.logger = logging.getLogger(f'LmpRun.{self.job_name}')
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers = []

        # Create file handler for logging to self.directory/lammps_run.log
        log_file = os.path.join(self.directory, 'lammps_run.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        self.logger.addHandler(fh)

        # Create console handler for warnings and errors
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(console_formatter)
        self.logger.addHandler(ch)

        # Handle lmp_script (string or file)
        if os.path.isfile(str(lmp_script)):
            with open(lmp_script, 'r') as f:
                self.lmp_script = f.read()
        else:
            self.lmp_script = lmp_script

        self.logger.info(f"LmpRun initialized for job: {job_name}")
        self.logger.info(f"Working directory: {self.directory}")

    def _check_existing_jobs(self):
        """Check for existing jobs in the working directory."""
        user = os.environ.get('USER')
        if not user:
            self.logger.error("Environment variable USER is not set.")
            return []

        try:
            result = subprocess.run(
                ['squeue', '-u', user, '--format=%i %Z'],
                capture_output=True, text=True, check=True
            )
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            return [parts[0] for parts in (line.strip().split() for line in lines)
                    if len(parts) >= 2 and os.path.abspath(parts[1]) == self.directory]
        except Exception as e:
            self.logger.error(f"Error checking existing jobs: {e}")
            return []

    def _check_completion(self):
        """Check if the LAMMPS calculation has completed."""
        try:
            log_file = os.path.join(self.directory, 'log.lammps')
            return os.path.exists(log_file) and 'Total wall time:' in open(log_file).read()
        except Exception as e:
            self.logger.error(f"Error checking completion: {e}")
            return False

    def prepare_files(self, cores=None, time=None):
        """
        Prepare all necessary files for LAMMPS calculation.

        Returns:
            bool: True if successful, False otherwise
        """
        cores = cores or self.default_cores
        time = time or self.default_time

        try:
            # Prepare directory
            os.makedirs(self.directory, exist_ok=True)

            # Write control.inp (LAMMPS script)
            with open(os.path.join(self.directory, 'control.inp'), 'w') as f:
                f.write(self.lmp_script)

            # Write structure.inp (atomic structure)
            structure_path = os.path.join(self.directory, 'structure.inp')
            write_lammps_data(structure_path, self.atoms,
                              specorder=self.element_order,
                              masses=True, atom_style='atomic')

            # Handle potential files
            if self.pot == 'ace':
                # Write potential.inp
                pot_content = (
                    f"# Potential file path: {self.pot_path}\n"
                    f"pair_style pace recursive\n"
                    f"pair_coeff * * {os.path.basename(self.pot_path)} {' '.join(self.element_order)}"
                )
                with open(os.path.join(self.directory, 'potential.inp'), 'w') as f:
                    f.write(pot_content)

                # Copy potential file
                if not self.pot_path or not os.path.exists(self.pot_path):
                    raise FileNotFoundError(f"Potential file not found at {self.pot_path}")
                shutil.copy2(self.pot_path,
                             os.path.join(self.directory, os.path.basename(self.pot_path)))

            # Write SLURM script
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
module load lammps_ml

srun lmp -in control.inp
"""
            with open(os.path.join(self.directory, 'submit.sh'), 'w') as f:
                f.write(sbatch_content)

            self.logger.info("All files prepared successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error preparing files: {e}")
            print(f"ERROR: {str(e)}")
            return False

    def write_files(self, time=None, cores=None):
        """
        Write all necessary files for LAMMPS calculation without job submission.

        Args:
            time (float, optional): Job runtime in hours
            cores (int, optional): Number of CPU cores

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            success = self.prepare_files(cores, time)
            if success:
                print(f"SUCCESS: All files written successfully to {self.directory}")
            return success
        except Exception as e:
            self.logger.error(f"Error in write_files: {e}")
            print(f"ERROR: Failed to write files - {str(e)}")
            return False

    def run(self, time=None, cores=None, rerun=False):
        """
        Prepare files and submit the LAMMPS calculation.

        Args:
            time (float, optional): Job runtime in hours
            cores (int, optional): Number of CPU cores
            rerun (bool, optional): Whether to rerun existing calculation

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check completion status
            if self._check_completion() and not rerun:
                print(f"WARNING: Calculation in {self.directory} has already completed.")
                return False

            # Handle existing jobs
            existing_jobs = self._check_existing_jobs()
            if existing_jobs:
                if not rerun:
                    print(f"WARNING: Job(s) {', '.join(existing_jobs)} already running in {self.directory}")
                    return False
                else:
                    for jobid in existing_jobs:
                        subprocess.run(['scancel', jobid], check=True)
                        self.logger.info(f"Cancelled job {jobid}")

            # Prepare all necessary files
            if not self.prepare_files(cores, time):
                return False

            # Submit job
            result = subprocess.run(
                ['sbatch', 'submit.sh'],
                check=True,
                cwd=self.directory,
                capture_output=True,
                text=True
            )

            # Get job ID from submission output
            job_id = result.stdout.strip().split()[-1]

            # Print job submission line
            print(f"{job_id}  {self.directory}")
            return True

        except Exception as e:
            self.logger.error(f"Error in job submission: {e}")
            print(f"ERROR: {str(e)}")
            return False

if __name__ == '__main__':
    pass

# Example usage
# Create LmpRun instance
# lmp_run = LmpRun(
#     lmp_script=lmp_script,
#     atoms=atoms,
#     directory='lmp_calc',
#     job_name='Cu_NPT',
#     pot='ace',
#     pot_path='/path/to/potential.yace',
#     element_order=['Cu']
# )

# # Just write files without submitting
# lmp_run.write_files(time=2.0, cores=24)


## TODO : One logger file for all the instances of LmpRun if it is same top directory
## TODO : make a logger file and log all details in that one and put in same directory.
## TODO : if lmprun.write() is called and all files are succesffuly written then print for each job else warning or error message
# job written to working_directory(top directory + job_name) 
## if lmprun.run() is called and then print
# id jobid working_directory for each job submitted else print error message or warning only.

