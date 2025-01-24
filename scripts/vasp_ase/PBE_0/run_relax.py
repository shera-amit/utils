from ase.io import read
from ase.calculators.vasp import Vasp

atoms = read('POSCAR', format='vasp')

### structure relax parameters
calc = Vasp(gga='PE',
            xc='pbe0',
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
            lhfcalc=True,
            algo='D',
            time=0.4,
            txt='-')

atoms.calc = calc
atoms.get_potential_energy()

