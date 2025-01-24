import os
from ase.io import read, write
from ase.db import connect

# Connect to the database
db = connect("./train.db")

# Iterate through all structures in the database
for row in db.select():
    # Create a directory with the ID of the structure
    dir_name = str(row.id)
    os.makedirs(dir_name, exist_ok=True)
    
    # Read the structure and write it as a POSCAR file
    atoms = row.toatoms()
    poscar_path = os.path.join(dir_name, "POSCAR")
    write(poscar_path, atoms, format="vasp", sort=True)



