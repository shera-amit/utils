from ase.constraints import ExpCellFilter, StrainFilter
from ase.db import connect
from ase.io import read
from ase.optimize import BFGS
from gpaw import GPAW, PW, Davidson, MixerSum

# Read the NiO structure from the CIF file
atoms = read('../NiO.cif', format='cif')

db = connect('NiO.db')

# 1. Optimize atomic positions

def calculator(encut, txt):
    return GPAW(mode=PW(encut),
                    xc='PBE',
                    mixer=MixerSum(0.1, 5, 100),
                    eigensolver=Davidson(3),
                    kpts=(13, 13, 13),
                    occupations={'name': 'fermi-dirac', 'width': 0.1},
                    txt=txt)

atoms.calc = calculator(1050, 'pos_relax.txt')
opt_pos = BFGS(atoms, logfile='pos_relax.log')
opt_pos.run(fmax=0.05)
db.write(atoms, relax='pos')

# 2. Optimize unit cell
# ddcut = ddedecut(atoms, 1050)
atoms.calc = calculator(1050, 'cell_relax.txt')
mask = [1, 1, 1, 0, 0, 0]
cell_filter = StrainFilter(atoms, mask=mask)
opt_cell = BFGS(cell_filter, logfile='cell_relax.log')
opt_cell.run(fmax=0.05)
db.write(atoms, relax='cell')

# again with new ddedecut
# 3. Optimize atomic positions and unit cell
# ddcut2 = ddedecut(atoms, 1050)
atoms.calc = calculator(1050, 'full_relax.txt')
both_filter = ExpCellFilter(atoms, mask=mask)
opt_both = BFGS(both_filter, logfile='full_relax.log')
opt_both.run(fmax=0.01)
db.write(atoms, relax='full')

atoms.write('NiO_final.cif', format='cif')
