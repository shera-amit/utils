# run.py
from ase.io import read
from ase.calculators.vasp import Vasp
import os
from ase.db import connect

atoms = read('POSCAR', format='vasp')
cwd = os.getcwd()
id = cwd.split('/')[-1]

### structure relax parameters
calc = Vasp(xc='pbe',
            prec='Accurate',
            encut=900,
            ediff=1e-6,
            ismear=0,
            sigma=0.1,
            ibrion=-1, # static calculation
            kspacing=0.20, # use supercell
            kgamma=True,
            ncore=8,
            lwave=False,
            lcharg=False,
            lreal=False,
            ialgo=38,
            txt='vasp.txt')

atoms.calc = calc
atoms.get_potential_energy()

db = connect('../train.db')
db.write(atoms, id=id)
