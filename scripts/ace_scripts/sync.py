from ase.db import connect
from ase.io import read

db = connect("db10_calculated.db")
for i in range(1, 601):
    atoms = read(f"./{i}/OUTCAR", format="vasp-out")
    db.write(atoms)
