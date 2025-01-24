from ase.io import read
from ase.calculators.vasp import Vasp

atoms = read('POSCAR', format='vasp')

### structure relax parameters
calc = Vasp(gga='pe',
            metagga='SCAN',
            prec='Accurate',
            encut=900,
            ediff=1e-6,
            ediffg=-0.5e-2,
            ismear=0,
            sigma=0.1,
            ibrion=1,
            kpts=(9, 9, 9),
            ncore=8,
            lwave=True,
            lcharg=True,
            lreal=False,
            nelm=60,
            nsw=100,
            isif=3,
            laechg=True,
            lasph=True,  #todo metagga calculations put this True
            txt='-')

atoms.calc = calc
atoms.get_potential_energy()

