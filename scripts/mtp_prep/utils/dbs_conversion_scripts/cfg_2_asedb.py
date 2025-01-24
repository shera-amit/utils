import ase.db
from ase import Atoms, Atom
import sys
import numpy as np
import argparse


def parse_arguments():
    description = 'Convert cfg data format to ase.db'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--cfg_filename', help='name of cfg file to be coverted', type=str, default="C-add_to_train.cfg")
    parser.add_argument('-so', '--species_order',  help='order of species in cfg file', type=str, default="Ba O Ti")
    parser.add_argument('-mv', '--max_atomic_volume',  help='maximum allowed atomic volume -- ignore structure otherwise ',
                        type=float, default=100)

    return parser.parse_args()


def process_cfg_line(line, type2X, max_atomic_volume, atoms, forces, stresses, energy, read_status, cell, db):
    if "BEGIN_CFG" in line:
        return Atoms([]), [], [], 0, {'size': False, 'supercell': False, 'atoms': False, 'energy': False, 'stresses': False}, []

    elif "END_CFG" in line:
        V = atoms.get_volume()
        if V / len(atoms) < max_atomic_volume:
            db.write(atoms, data={"dft_forces": np.array(forces),
                                   "dft_energy": energy,
                                   "dft_stresses": stresses})
        else:
            print("Volume too large, V_per_atom = %.2f" % (V / len(atoms)))

    elif read_status['size']:
        size = int(line.strip())
        read_status['size'] = False

    elif read_status['supercell']:
        read_status['supercell'] = process_supercell_line(line, cell, atoms)

    elif read_status['atoms']:
        read_status['atoms'], forces = process_atom_line(line, type2X, atoms, forces)

    elif read_status['energy']:
        energy = float(line.strip())
        read_status['energy'] = False

    elif read_status['stresses']:
        stresses = np.array(line.strip().split(), dtype=float)
        read_status['stresses'] = False

    else:
        read_status = update_read_status(line, read_status)

    return atoms, forces, stresses, energy, read_status, cell


def process_supercell_line(line, cell, atoms):
    vec = np.array(line.strip().split(), dtype=float)
    cell.append(vec)

    if len(cell) > 3:
        atoms.set_cell(cell)
        atoms.pbc = True
        return False

    return True


def process_atom_line(line, type2X, atoms, forces):
    tmp = line.strip().split()
    type_id = tmp[1]
    position = np.array(tmp[2:5], dtype=float)
    at = Atom(type2X[int(type_id)], position)
    atoms.append(at)

    force = np.array(tmp[5:8], dtype=float)
    forces.append(force)

    return False, forces


def update_read_status(line, read_status):
    if "Size" in line:
        read_status['size'] = True
    elif "AtomData" in line:
        read_status['atoms'] = True
    elif "Supercell" in line:
        read_status['supercell'] = True
    elif "Energy" in line:
        read_status['energy'] = True
    elif "PlusStress" in line:
        read_status['stresses'] = True

    return read_status


def main():
   
    args = parse_arguments()

    asedb = args.cfg_filename.strip("cfg") + "db"
    db = ase.db.connect(asedb)
    specorder = args.species_order.split()
    indices = list(range(len(specorder)))
    type2X = dict(zip(indices, specorder))

    atoms = Atoms([])
    forces = []
    stresses = []
    energy = 0
    read_status = {'size': False, 'supercell': False, 'atoms': False, 'energy': False, 'stresses': False}
    cell = []

    with open(args.cfg_filename, "r") as f:
        for line in f.readlines():
            atoms, forces, stresses, energy, read_status, cell = process_cfg_line(line, type2X, args.max_atomic_volume,
                                                                                   atoms, forces, stresses, energy,
                                                                                   read_status, cell, db)

    print(f"Conversion complete. {asedb} created.")


if __name__ == "__main__":
    main()
