#! /home/as41vomu//miniconda3/bin python

import argparse
import shutil
from pathlib import Path

# Constants for directories
UTILS_DIR = Path("/home/as41vomu/scripts/mtp_prep/utils")
SUB_DIR = "fit_0"
FIT_SUB_DIRS = ["active_learning", "dbs", "dft_calculations", "mtp_fit"]
DBS_SUB_DIRS = ["initial_db", "active_learning_db"]
DFT_SUB_DIRS = ["static_initial", "active_learning"]
SCRIPT_DIRS = {
    "dbs": "dbs_conversion_scripts",
    "dft_calculations": "dft_calculations_scripts",
    "active_learning": "active_learning_scripts",
    "mtp_fit": "mtp_fit_scripts",
}

def copy_all_files(src_dir: Path, dst_dir: Path):
    """Copy all files from src_dir to dst_dir."""
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy(item, dst_dir)

def create_dir_and_copy_scripts(parent_dir: Path, script_dir: str):
    """Creates a directory and copies scripts into it."""
    parent_dir.mkdir(parents=True, exist_ok=True)
    copy_all_files(UTILS_DIR / script_dir, parent_dir)

def prep_dir(root_dir: str="test"):
    """Prepares directory structure under the given root directory."""
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory
    sub_dir = root_dir / SUB_DIR

    # Iterate over fit subdirectories
    for fit_sub_dir_name in FIT_SUB_DIRS:
        fit_sub_dir = sub_dir / fit_sub_dir_name

        if fit_sub_dir_name == "dbs":
            for dbs_sub_dir_name in DBS_SUB_DIRS:
                dbs_sub_dir = fit_sub_dir / dbs_sub_dir_name
                create_dir_and_copy_scripts(dbs_sub_dir, SCRIPT_DIRS["dbs"])

        elif fit_sub_dir_name == "dft_calculations":
            for dft_sub_dir_name in DFT_SUB_DIRS:
                dft_sub_dir = fit_sub_dir / dft_sub_dir_name
                create_dir_and_copy_scripts(dft_sub_dir, SCRIPT_DIRS["dft_calculations"])

        else:
            create_dir_and_copy_scripts(fit_sub_dir, SCRIPT_DIRS[fit_sub_dir_name])

def main():
    parser = argparse.ArgumentParser(description="Prepare directory structure.")
    parser.add_argument("root_dir", help="Root directory for the structure.")
    args = parser.parse_args()
    prep_dir(args.root_dir)

if __name__ == "__main__":
    main()

