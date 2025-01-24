import os
import logging
import sqlite3
import subprocess
from datetime import datetime
from ase.calculators.vasp import Vasp
import re
from datetime import datetime
from contextlib import contextmanager
import pandas as pd

class Project:
    def __init__(self, project_dir):
        self.project_dir = os.path.abspath(project_dir)
        os.makedirs(self.project_dir, exist_ok=True)
        self.db_path = os.path.join(os.path.expanduser('~'), 'project.db')
        self.logger = self.setup_logger()
        self.ensure_database()
        self.logger.info(f"Project initialized at {self.project_dir}")

    def setup_logger(self):
        logger = logging.getLogger(f'ProjectLogger_{self.project_dir}')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(os.path.join(self.project_dir, 'project.log'))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger
    
    def ensure_database(self):
        if not os.path.exists(self.db_path):
            self.initialize_database()
        else:
            self.logger.info("Database already exists. Skipping initialization.")
    
    @contextmanager
    def get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def initialize_database(self):
        self.logger.info("Initializing database...")
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_dir TEXT,
                    job_name TEXT,
                    job_directory TEXT,
                    status TEXT,
                    slurm_job_id TEXT,
                    start_time TEXT,
                    end_time TEXT
                )
            ''')
            conn.commit()
        self.logger.info("Database initialized successfully.")

    def add_job_to_db(self, job):
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO jobs (project_dir, job_name, job_directory, status, slurm_job_id, start_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.project_dir, job.job_name, job.job_directory, job.status, job.slurm_job_id, datetime.now().isoformat()))
            conn.commit()

    def update_job_status(self, job_name, status, start_time=None, end_time=None, slurm_job_id=None):
        updates = []
        params = []
        if status:
            updates.append('status = ?')
            params.append(status)
        if start_time:
            updates.append('start_time = ?')
            params.append(start_time)
        if end_time:
            updates.append('end_time = ?')
            params.append(end_time)
        if slurm_job_id:
            updates.append('slurm_job_id = ?')
            params.append(slurm_job_id)
        if updates:
            update_str = ', '.join(updates)
            params.extend([job_name, self.project_dir])
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    UPDATE jobs
                    SET {update_str}
                    WHERE job_name = ? AND project_dir = ?
                ''', params)
                conn.commit()


    def refresh_job_statuses(self):
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT job_name, slurm_job_id, status 
                    FROM jobs 
                    WHERE project_dir = ? AND slurm_job_id IS NOT NULL
                ''', (self.project_dir,))
                rows = cursor.fetchall()

            updates = []
            for job_name, slurm_job_id, status in rows:
                try:
                    result = subprocess.run(
                        ['sacct', '-j', slurm_job_id, '--format=JobID,State,Start,End', '--noheader', '--parsable2'],
                        capture_output=True, text=True, check=True
                    )
                    
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines:
                        parts = line.strip().split('|')
                        if len(parts) >= 4:
                            job_id, state, start_time, end_time = parts[:4]
                            
                            if job_id == slurm_job_id or job_id.startswith(f'{slurm_job_id}.'):
                                start_time = None if start_time in ('Unknown', '') else start_time
                                end_time = None if end_time in ('Unknown', '') else end_time
                                
                                updates.append((state, start_time, end_time, job_name, self.project_dir))
                                break
                    else:
                        self.logger.warning(f"No matching job found for SLURM job ID: {slurm_job_id}")
                
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error querying sacct for job {job_name}: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error refreshing status for job {job_name}: {e}")

            # Batch update job statuses
            if updates:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.executemany('''
                        UPDATE jobs
                        SET status = ?, start_time = ?, end_time = ?
                        WHERE job_name = ? AND project_dir = ?
                    ''', updates)
                    conn.commit()
        
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in refresh_job_statuses: {e}")

    def job_table(self, status="", job_name="", refresh=False, limit=20):
        if refresh:
            self.refresh_job_statuses()
        
        query = '''
            SELECT DISTINCT job_name, slurm_job_id, status, start_time, end_time
            FROM jobs 
            WHERE project_dir = ?
        '''
        params = [self.project_dir]
        if status:
            query += ' AND status LIKE ?'
            params.append(f'%{status}%')
        if job_name:
            query += ' AND job_name LIKE ?'
            params.append(f'%{job_name}%')
        query += ' ORDER BY start_time DESC LIMIT ?'
        params.append(limit)
        
        with self.get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            print("No jobs found.")
            return None

        # Format datetime columns
        for col in ['start_time', 'end_time']:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%b %d, %Y %I:%M %p')

        # Replace NaN with 'N/A'
        df = df.fillna('N/A')

        # Rename columns for display
        df.columns = ['Job Name', 'Slurm Job ID', 'Status', 'Start Time', 'End Time']

        # Display the table
        # print(df.to_string(index=False))
        
        if len(df) == limit:
            print(f"\nNote: Showing only the {limit} most recent jobs. Use a higher 'limit' parameter to see more.")
        
        return df
    


    def get_job(self, job_name):
        jobs = self.job_table(job_name=job_name)
        return jobs[0] if jobs else None

class Job:
    def __init__(self, project, job_name, calc, atoms, account='p0020994'):
        self.project = project
        self.job_name = job_name
        self.calc = calc
        self.atoms = atoms
        self.account = account
        self.job_directory = os.path.join(self.project.project_dir, self.job_name)
        os.makedirs(self.job_directory, exist_ok=True)
        self.logger = self.setup_logger()
        self.slurm_job_id = None
        self.status = 'pending'
        self.start_time = None
        self.end_time = None
        self.project.add_job_to_db(self)
        self.logger.info(f"Job '{self.job_name}' initialized in project '{self.project.project_dir}'")

    @classmethod
    def from_db_row(cls, project, row):
        job_id, project_dir, job_name, job_directory, status, slurm_job_id, start_time, end_time = row
        job = cls(project, job_name, calc=None, atoms=None)
        job.job_directory = job_directory
        job.status = status
        job.slurm_job_id = slurm_job_id
        job.start_time = start_time
        job.end_time = end_time
        return job

    def setup_logger(self):
        logger = logging.getLogger(f'JobLogger_{self.job_name}')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(os.path.join(self.job_directory, 'job.log'))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger

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
        with open(os.path.join(self.job_directory, 'submit.sh'), 'w') as f:
            f.write(sbatch_content)
        self.logger.info(f"SBATCH file written to {os.path.join(self.job_directory, 'submit.sh')}")

    def check_existing_jobs(self):
        user = os.environ.get('USER')
        if not user:
            self.logger.error("Environment variable USER not set.")
            return []
        try:
            result = subprocess.run(['squeue', '-u', user, '--format=%i %Z'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            lines = lines[1:]  # Skip header
            jobs_in_directory = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    jobid = parts[0]
                    job_dir = parts[1]
                    if os.path.abspath(job_dir) == self.job_directory:
                        jobs_in_directory.append(jobid)
            self.logger.info(f"Existing jobs in directory: {', '.join(jobs_in_directory) if jobs_in_directory else 'None'}")
            return jobs_in_directory
        except Exception as e:
            self.logger.error(f"Error checking existing jobs: {e}")
            return []

    def check_convergence(self, outcar_path):
        try:
            with open(outcar_path, 'r') as f:
                lines = f.readlines()[-100:]  # Read last 100 lines
            content = ''.join(lines)
            converged = 'reached required accuracy' in content
            self.logger.info(f"Convergence check: {'Converged' if converged else 'Not converged'}")
            return converged
        except Exception as e:
            self.logger.error(f"Error checking convergence in {outcar_path}: {e}")
            return False

    def run(self, cores=96, time=24.0, rerun=False):
        self.logger.info(f"Run initiated with time={time}, cores={cores}, rerun={rerun}")
        outcar_path = os.path.join(self.job_directory, 'OUTCAR')

        if os.path.exists(outcar_path):
            if self.check_convergence(outcar_path):
                if not rerun:
                    message = f"Calculation in {self.job_directory} has already converged. Not submitting a new job."
                    self.logger.warning(message)
                    print(f"WARNING: {message}")
                    print("To rerun the calculation, use rerun=True")
                    return
                else:
                    self.logger.warning(f"Rerunning the calculation in {self.job_directory} despite convergence.")

        existing_jobs = self.check_existing_jobs()
        if existing_jobs and not rerun:
            message = f"Job(s) {', '.join(existing_jobs)} from {self.job_directory} already pending or running. Not submitting a new job."
            self.logger.warning(message)
            print(f"WARNING: {message}")
            print("To cancel and resubmit the job, use rerun=True")
            return
        elif existing_jobs and rerun:
            for jobid in existing_jobs:
                try:
                    subprocess.run(['scancel', jobid], check=True)
                    message = f"Cancelled job {jobid} from directory {self.job_directory}"
                    self.logger.info(message)
                except subprocess.CalledProcessError as e:
                    error_message = f"Error cancelling job {jobid}: {e}"
                    self.logger.error(error_message)
                    print(f"ERROR: {error_message}")

        # Prepare input files
        if self.calc and self.atoms:
            self.calc.directory = self.job_directory
            self.atoms.set_calculator(self.calc)
            self.calc.write_input(self.atoms)
            self.logger.info("VASP input files written")
        else:
            self.logger.error("Calculator and atoms must be provided to write input files.")
            return

        # Write sbatch file
        self.write_sbatch_file(cores, time)

        # Submit job
        try:
            result = subprocess.run(['sbatch', 'submit.sh'], check=True, cwd=self.job_directory, capture_output=True, text=True)
            output = result.stdout.strip()
            job_id = output.split()[-1]
            message = f"Job ID: {job_id}, Working directory: {os.path.abspath(self.job_directory)}"
            self.logger.info(message)
            print(message)
            self.slurm_job_id = job_id
            self.status = 'RUNNING'
            self.project.update_job_status(self.job_name, self.status, slurm_job_id=job_id)
        except subprocess.CalledProcessError as e:
            error_message = f"Error in job submission: {e}"
            self.logger.error(error_message)
            print(f"ERROR: {error_message}")

# Usage Example

if __name__ == "__main__":
    pass
