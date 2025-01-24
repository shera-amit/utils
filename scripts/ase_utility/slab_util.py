from ase.constraints import FixAtoms

def _tag_atoms(slab, tol=1e-3):
    """
    (Private function)
    Assigns a tag to each atom in the slab based on its z-coordinate.
    """
    sorted_atoms = sorted(slab, key=lambda atom: atom.position[2])
    current_tag = 1
    current_z = sorted_atoms[0].position[2]
    
    for atom in sorted_atoms:
        if abs(atom.position[2] - current_z) > tol:
            current_tag += 1
            current_z = atom.position[2]
        atom.tag = current_tag

def constrain_layers(slab, n=3, tol=1e-3):
    """
    Constrain all atoms in the slab except for the top and bottom n layers.
    """
    # First tag atoms by layer
    _tag_atoms(slab, tol)

    all_tags = sorted(list(set([atom.tag for atom in slab])))
    mobile_tags = all_tags[:n] + all_tags[-n:]

    fixed_indices = [atom.index for atom in slab if atom.tag not in mobile_tags]
    slab.set_constraint(FixAtoms(indices=fixed_indices))

