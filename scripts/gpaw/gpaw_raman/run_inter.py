import numpy as np
from ase.build import bulk
from ase.phonons import Phonons
from ase.units import invcm
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.elph.electronphonon import ElectronPhononCoupling
from gpaw.raman.elph import EPC
from gpaw.raman.dipoletransition import get_momentum_transitions
from gpaw.raman.raman import (
    calculate_raman, calculate_raman_intensity, plot_raman)
from ase.db import connect
from ase.io import read


def calculate_elph_potential(atoms):
    calc = GPAW(mode='lcao', basis='dzp',
                kpts=(4, 4, 4), xc='PBE',
                symmetry={'point_group': False},
                txt='elph.txt')
    elph = ElectronPhononCoupling(atoms, calc, supercell=(2, 2, 2),
                                  calculate_forces=False)
    elph.run()


def calculate_supercell_matrix(atoms, supercell):
    atoms_sc = atoms * supercell
    calc = GPAW(mode='lcao', basis='dzp',
                kpts=(6, 6, 6), xc='PBE',
                symmetry={'point_group': False},
                convergence={'bands': 'nao', 'density': 1e-5},
                parallel={'domain': 1},
                txt='gs_super.txt')
    atoms_sc.calc = calc
    atoms_sc.get_potential_energy()

    elph = EPC(atoms, supercell=supercell)
    elph.calculate_supercell_matrix(calc, include_pseudo=True)


def calculate_momentum_matrix(atoms):
    calc = GPAW(mode='lcao', basis='dzp',
                kpts=(6, 6, 6), xc='PBE',
                symmetry={'point_group': False},
                convergence={'bands': 'nao', 'density': 1e-5},
                parallel={'band': 1},
                txt='mom.txt')

    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.calc.write("gs.gpw", mode="all")
    get_momentum_transitions(calc.wfs, savetofile=True)


def main():
    #db = connect('NiO.db')
    atoms = read('NiO.cif', format='cif') 

    atoms2 = atoms.copy()
    # Calculate electron-phonon potential
    calculate_elph_potential(atoms2)

    # Calculate supercell matrix
    supercell = (2, 2, 2)
    atoms2 = atoms.copy()
    calculate_supercell_matrix(atoms2, supercell)

    # Calculate momentum matrix
    atoms2 = atoms.copy()
    calculate_momentum_matrix(atoms2)


if __name__ == '__main__':
    main()
