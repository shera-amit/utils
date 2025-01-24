import os
import numpy as np
from ase.io.lammpsdata import write_lammps_data
from mp_api.client import MPRester


def strain_cell(atoms, normal=None, shear=None):
    """
    Apply normal strain and shear strain to the cell of an ASE Atoms object.

    Parameters:
    - atoms (Atoms): The ASE Atoms object to be strained.
    - normal (float or list, optional): The magnitude(s) of normal strain to apply.
      If a float, apply the same normal strain in all directions.
      If a list of length 3, apply different normal strains in x, y, and z directions.
    - shear (float or tuple, optional): The magnitude(s) of shear strain to apply.
      If a float, apply the same shear strain to all shear components (xy, yz, zx).
      If a tuple of length 3, apply different shear strains to xy, yz, and zx components.

    Returns:
    - strained_atoms (Atoms): The strained ASE Atoms object.
    """
    cell = atoms.get_cell()
    strain_matrix = np.eye(3)

    if normal is not None:
        if isinstance(normal, (float, int)):
            strain_matrix += np.eye(3) * normal
        elif isinstance(normal, (list, tuple)) and len(normal) == 3:
            strain_matrix[0, 0] += normal[0]  # Normal strain in x
            strain_matrix[1, 1] += normal[1]  # Normal strain in y
            strain_matrix[2, 2] += normal[2]  # Normal strain in z
        else:
            raise ValueError("Invalid input for normal strain. Must be a float or a list of length 3.")

    if shear is not None:
        if isinstance(shear, (float, int)):
            strain_matrix[0, 1] = strain_matrix[1, 0] = shear / 2 # Shear strain xy
            strain_matrix[1, 2] = strain_matrix[2, 1] = shear  / 2# Shear strain yz
            strain_matrix[2, 0] = strain_matrix[0, 2] = shear  / 2# Shear strain zx
        elif isinstance(shear, (list, tuple)) and len(shear) == 3:
            xy, yz, zx = shear
            strain_matrix[0, 1] = strain_matrix[1, 0] = xy / 2
            strain_matrix[1, 2] = strain_matrix[2, 1] = yz / 2
            strain_matrix[2, 0] = strain_matrix[0, 2] = zx /2 
        else:
            raise ValueError("Invalid input for shear strain. Must be a float or a tuple of length 3.")

    strained_cell = np.dot(strain_matrix, cell)
    strained_atoms = atoms.copy()
    strained_atoms.set_cell(strained_cell, scale_atoms=True)

    return strained_atoms

# Set your Materials Project API key
mp_key = os.environ.get("mp_key")

# Define the material IDs
mids = ['mp-2998', 'mp-5020', 'mp-5777', 'mp-5986']

# Fetch the structures from Materials Project
atoms_list = []
for mid in mids:
    with MPRester(mp_key) as mpr:
        if mid == 'mp-2998':
            structure = mpr.get_structure_by_material_id(mid, conventional_unit_cell=True)
        else:
            structure = mpr.get_structure_by_material_id(mid)
        atoms = structure.to_ase_atoms()
        atoms_list.append(atoms)

# Create supercells
suatoms_list = []
for atom in atoms_list:
    suatoms = atom.copy()
    suatoms.set_pbc([True, True, True])
    suatoms = suatoms.repeat((2, 2, 2))
    suatoms.center()
    suatoms.wrap()
    suatoms_list.append(suatoms)

# Define the LAMMPS script

lmp_script = r"""
units       metal
boundary    p p p
atom_style  atomic
box tilt large



read_data    structure.inp

mass 1      137.327
mass 2      15.999
mass 3      47.867



pair_style pace/extrapolation
pair_coeff * * ../../fit/output_potential.yaml ../../fit/output_potential.asi Ba O Ti


# define variables
variable pr equal 0.0
variable seed equal 83948
variable td equal $(100.0*dt)
variable pd equal $(1000.0*dt)
thermo 50
timestep 0.001

fix data1 all print 1000 "$(step) $(temp) $(pxx) $(pyy) $(pzz) $(press) $(etotal) $(pe) $(enthalpy) $(vol) $(density)" append thermo_data.txt screen no

# pace setting
fix pace_gamma all pair 10 pace/extrapolation gamma 1
compute max_pace_gamma all reduce max f_pace_gamma

variable dump_skip equal "c_max_pace_gamma < 5"

dump pace_dump all custom 200 extrapolative_structures.dump id type x y z f_pace_gamma
dump_modify pace_dump skip v_dump_skip

variable max_pace_gamma equal c_max_pace_gamma
fix extreme_extrapolation all halt 10 v_max_pace_gamma > 25


velocity all create 100 ${seed} 
#starting simulation
fix 1 all npt temp 100 100 ${td} x ${pr} ${pr} ${pd} y ${pr} ${pr} ${pd} z ${pr} ${pr} ${pd} xy ${pr} ${pr} ${pd} xz ${pr} ${pr} ${pd} yz ${pr} ${pr} ${pd}
run 100000
unfix 1

fix 2 all npt temp 100 2000 ${td} x ${pr} ${pr} ${pd} y ${pr} ${pr} ${pd} z ${pr} ${pr} ${pd} xy ${pr} ${pr} ${pd} xz ${pr} ${pr} ${pd} yz ${pr} ${pr} ${pd}
run 500000
unfix 2

fix 3 all npt temp 2000 2000 ${td} x ${pr} ${pr} ${pd} y ${pr} ${pr} ${pd} z ${pr} ${pr} ${pd} xy ${pr} ${pr} ${pd} xz ${pr} ${pr} ${pd} yz ${pr} ${pr} ${pd}
run 100000
unfix 3

fix 4 all npt temp 2000 100 ${td} x ${pr} ${pr} ${pd} y ${pr} ${pr} ${pd} z ${pr} ${pr} ${pd} xy ${pr} ${pr} ${pd} xz ${pr} ${pr} ${pd} yz ${pr} ${pr} ${pd}
run 500000
unfix 4

write_data finished.data
"""

# Apply strains to the structures
normal_strain = np.linspace(-0.1, 0.1, 5)
shear_strain = np.linspace(-0.03, 0.03, 5)
final_atoms_list = []
for i, atoms in enumerate(suatoms_list):
    for ns in normal_strain:
        strained_atoms = strain_cell(atoms, normal=ns)
        final_atoms_list.append(strained_atoms)
    for ss in shear_strain:
        strained_atoms = strain_cell(atoms, shear=ss)
        final_atoms_list.append(strained_atoms)

# Set up pressure and random seed values
pressure = np.linspace(0, 50, 6)
random_seed = np.random.randint(0, 100000, 5)
dir_counter = 1

# Create directories and run LAMMPS simulations
for i, atoms in enumerate(final_atoms_list):
    for p in pressure:
        for s in random_seed:
            pr = int(p)
            seed = int(s)
            work_dir = str(dir_counter)
            os.makedirs(work_dir, exist_ok=True)
            write_lammps_data(f'{work_dir}/structure.inp', atoms, specorder=['Ba', 'O', 'Ti'])
            modified_lmp_script = lmp_script.replace('variable seed equal 83948', f'variable seed equal {seed}')
            modified_lmp_script = modified_lmp_script.replace('variable pr equal 0.0', f'variable pr equal {pr}')
            with open(f"{work_dir}/control.inp", "w") as f:
                f.write(modified_lmp_script)
            dir_counter += 1
