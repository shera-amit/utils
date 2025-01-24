import numpy as np
from ase.build import bulk
from ase.phonons import Phonons
from ase.units import invcm
from gpaw import GPAW
from gpaw.mpi import world
from ase.io import read

atoms = read('./NiO.cif', format='cif')
calc = GPAW(mode={'name': 'pw', 'ecut': 850},
            kpts=(6, 6, 6), xc='PBE',
            symmetry={'point_group': False},
            convergence={'density': 0.5e-5},
            txt='phonons.txt')

# Phonon calculation
ph_supercell = (2, 2, 2)
ph = Phonons(atoms, calc, supercell=ph_supercell, delta=0.01)
ph.run()

# To display results (optional)
ph.read(method='frederiksen', acoustic=True)
frequencies = ph.band_structure([[0, 0, 0], ])[0]  # Get frequencies at Gamma
if world.rank == 0:
    # save phonon frequencies for later use
    np.save("vib_frequencies.npy", frequencies)
