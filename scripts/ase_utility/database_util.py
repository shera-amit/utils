import numpy as np


def strain_atoms(atoms, strain_components):
    """
    Apply strain to an ASE Atoms object.

    Parameters:
    - atoms: ASE Atoms object
    - strain_components: List of strain components [xx, yy, zz, xy, yz, zx]

    Returns:
    - Strained Atoms object
    """
    # Extract strain components
    atoms = atoms.copy()
    xx, yy, zz, xy, yz, zx = strain_components

    # Create the strain matrix
    strain_matrix = np.array(
        [
            [1 + xx, 0.5 * xy, 0.5 * zx],
            [-0.5 * xy, 1 + yy, 0.5 * yz],
            [-0.5 * zx, -0.5 * yz, 1 + zz],
        ]
    )

    # strain_matrix = np.array([[1 + xx, xy, zx],
    #                           [-xy, 1 + yy, yz],
    #                           [-zx, -yz, 1 + zz]])
    # Get the current cell vectors
    cell = atoms.get_cell()

    # Apply strain to the cell vectors
    strained_cell = np.dot(strain_matrix, cell)

    # Update the cell vectors of the Atoms object
    atoms.set_cell(strained_cell, scale_atoms=True)

    return atoms


def get_distance_matrix(atoms):
    """
    Calculate the distance matrix for an ASE Atoms object.

    Parameters:
    - atoms: ASE Atoms object

    Returns:
    - distance_matrix: NumPy array of shape (natoms, natoms) containing the distances between atoms
    and min distance between set of atoms in structure.
    """
    # Get the positions of all atoms
    positions = atoms.get_positions()

    # Calculate the squared distances between all pairs of atoms
    dist_squared = np.sum(
        (positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2, axis=-1
    )

    # Take the square root to get the actual distances
    distance_matrix = np.sqrt(dist_squared)
    min_distance = np.min(distance_matrix[np.nonzero(distance_matrix)])

    return distance_matrix, min_distance


def perturb(atoms, min_distance, max_distance):
    """
    Add a random vector with a distance between min_distance and max_distance to the positions of atoms.

    Parameters:
    - atoms: ASE Atoms object
    - min_distance: Minimum distance of the random vector
    - max_distance: Maximum distance of the random vector

    Returns:
    - atoms: Updated ASE Atoms object with new positions
    """
    # Get the number of atoms
    natoms = len(atoms)

    # Generate random vectors for each atom
    random_vectors = np.random.rand(natoms, 3)
    random_vectors /= np.linalg.norm(random_vectors, axis=1)[:, np.newaxis]

    # Generate random distances for each atom
    distances = np.random.uniform(min_distance, max_distance, size=natoms)

    # Scale the random vectors by the distances
    random_vectors *= distances[:, np.newaxis]

    # Add the random vectors to the atom positions
    atoms.positions += random_vectors

    return atoms
