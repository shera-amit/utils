#!/work/scratch/as41vomu/mambaforge/envs/pyiron/bin/python
from ase.io import read
import numpy as np
from ase.units import Rydberg
from termcolor import colored

# Read atoms from the file
atoms_list = read("./scf.out", format="espresso-out", index=":")

# Extract potential energy and forces
energy_ev = np.array([atoms.get_potential_energy() for atoms in atoms_list])
forces = [np.max(np.linalg.norm(atoms.get_forces(), axis=1)) for atoms in atoms_list]

# Convert energy from eV to Ry
energy_ry = energy_ev / Rydberg

# Calculate energy differences between successive iterations
energy_diff_ev = np.diff(energy_ev)
energy_diff_ry = energy_diff_ev / Rydberg


# Function to print colored headers
def print_header(*headers):
    header_line = "".join(
        f"{colored(header, 'blue', attrs=['bold']):>20}" for header in headers
    )
    print(header_line)
    print("=" * len(header_line))


# Function to print colored data rows
def print_row(iteration, energy, energy_diff, force, units):
    if energy_diff == "N/A":
        energy_diff_str = f"{energy_diff:>20}"
    else:
        energy_diff_str = f"{energy_diff:20.6e}"
    row = f"{iteration:>10} {energy:20.6e} {energy_diff_str} {force:20.6e} {units}"
    print(row)


# Print the results in eV
print(colored("Table in Electron Volts (eV)", "green", attrs=["bold"]))
print_header("Iteration", "Energy(eV)", "    Energy Diff(eV)", "    Max Force(eV/Å)")
for i in range(len(energy_ev)):
    if i == 0:
        print_row(i, energy_ev[i], "N/A", forces[i], "")
    else:
        print_row(i, energy_ev[i], energy_diff_ev[i - 1], forces[i], "")

print("\n" + colored("Table in Atomic Units (Rydberg)", "green", attrs=["bold"]))
# Print the results in Ry
print_header("Iteration", "Energy(Ry)", "    Energy Diff(Ry)", "    Max Force(Ry/Bohr)")
for i in range(len(energy_ry)):
    if i == 0:
        print_row(i, energy_ry[i], "N/A", forces[i] / Rydberg, "")
    else:
        print_row(i, energy_ry[i], energy_diff_ry[i - 1], forces[i] / Rydberg, "")
