# ase_db_merge.py

from ase.db import connect

def merge_databases(db_paths, merged_db_path):
    """
    Merges multiple ASE databases into one.

    Parameters:
    db_paths (list of str): Paths to the databases to be merged.
    merged_db_path (str): Path to the output merged database.
    """
    with connect(merged_db_path) as merged_db:
        for db_path in db_paths:
            with connect(db_path) as db:
                for record in db.select():
                    merged_db.write(record)

