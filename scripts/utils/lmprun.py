import os
import logging
import subprocess
from ase.io.lammpsdata import write_lammps_data
import shutil
from datetime import datetime

class LmpRun:
    _loggers = {}  # Class-level dictionary to store loggers per top directory
    _submission_id = 1  # Class-level submission ID counter

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

        # Compute the top directory and job directory
        self.top_directory = os.path.abspath(directory or '.')
        self.directory = os.path.abspath(os.path.join(self.top_directory, job_name))
        os.makedirs(self.directory, exist_ok=True)

        # Configure logging for this instance
        if self.top_directory in LmpRun._loggers:
            self.logger = LmpRun._loggers[self.top_directory]
        else:
            # Create a new logger for the top directory
            logger_name = f'LmpRun.{os.path.basename(self.top_directory)}'
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.INFO)
            # Prevent propagation to avoid duplicate logging
            self.logger.propagate = False
            # Remove existing handlers
            self.logger.handlers = []
            
            # Create file handler for logging to self.top_directory/lammps_run.log
            log_file = os.path.join(self.top_directory, 'lammps_run.log')
            fh = logging.FileHandler(log_file)
            # Create a filter to only allow INFO level messages to file
            class InfoFilter(logging.Filter):
                def filter(self, record):
                    return record.levelno == logging.INFO
            
            fh.addFilter(InfoFilter())
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(file_formatter)
            self.logger.addHandler(fh)
            
            # Create console handler for warnings and errors only
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            # Create a filter to only allow WARNING and ERROR messages to console
            class WarningErrorFilter(logging.Filter):
                def filter(self, record):
                    return record.levelno >= logging.WARNING
                    
            ch.addFilter(WarningErrorFilter())
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            ch.setFormatter(console_formatter)
            self.logger.addHandler(ch)
            
            # Store logger in class variable
            LmpRun._loggers[self.top_directory] = self.logger

        # Handle lmp_script (string or file)
        if os.path.isfile(str(lmp_script)):
            with open(lmp_script, 'r') as f:
                self.lmp_script = f.read()
        else:
            self.lmp_script = lmp_script

        # Log initialization details to file only
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

rm -rf $HOME/.cache/lmod/* ; chmod 0500 $HOME/.cache/lmod
module purge
module load lammps_intel/2024.10.31

srun lmp -in control.inp
"""
            with open(os.path.join(self.directory, 'submit.sh'), 'w') as f:
                f.write(sbatch_content)

            self.logger.info(f"All files prepared successfully for job: {self.job_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error preparing files: {e}")
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
                print(f"Files written to {self.directory}")
                self.logger.info(f"Files successfully written to {self.directory}")
            return success
        except Exception as e:
            self.logger.error(f"Error in write_files: {e}")
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
                self.logger.warning(f"Calculation in {self.directory} has already completed.")
                return False

            # Handle existing jobs
            existing_jobs = self._check_existing_jobs()
            if existing_jobs:
                if not rerun:
                    self.logger.warning(f"Job(s) {', '.join(existing_jobs)} already running in {self.directory}")
                    return False
                else:
                    for jobid in existing_jobs:
                        subprocess.run(['scancel', jobid], check=True)
                        self.logger.info(f"Cancelled job {jobid}")

            # Prepare all necessary files
            if not self.prepare_files(cores, time):
                self.logger.error(f"Failed to prepare files for job {self.job_name}")
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

            # Get submission ID
            submission_id = LmpRun._submission_id
            
            # Log submission details to file only
            self.logger.info(f"Job submitted successfully - ID: {job_id}, Directory: {self.directory}")
            
            # Print only the submission line to console
            import sys
            sys.stdout.write(f"{submission_id}  {job_id}  {self.directory}\n")
            sys.stdout.flush()
            
            # Increment submission ID counter
            LmpRun._submission_id += 1

            return True

        except Exception as e:
            self.logger.error(f"Error in job submission: {e}")
            return False

if __name__ == '__main__':
    pass
