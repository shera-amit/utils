def analyze_ase_database(
    db_path, 
    output_dir=None, 
    show_plots=True,
    dpi=300
):
    """
    Analyze an ASE database and produce a set of commonly useful plots.
    
    Parameters
    ----------
    db_path : str
        Path to the ASE database file.
    output_dir : str, optional
        Directory to save analysis results. If None, plots won't be saved.
    show_plots : bool, optional
        Whether to display plots interactively (default: True).
    dpi : int, optional
        Resolution of the saved figure (default: 300).
        
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing per-structure statistics.
    """
    import ase.db
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from tqdm import tqdm
    
    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Connect to database
    db = ase.db.connect(db_path)
    
    results = []
    all_forces = []  # Store all forces from all atoms
    
    # First, verify we have at least one row
    row_count = len(list(db.select()))
    if row_count == 0:
        raise ValueError("Database is empty")
        
    for row in tqdm(db.select(), desc="Analyzing structures", total=row_count):
        atoms = row.toatoms()
        
        # Calculate minimum interatomic distance
        positions = atoms.get_positions()
        dist_matrix = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix)
        
        # Calculate energy per atom
        energy_per_atom = np.nan
        if hasattr(row, 'energy') and row.energy is not None:
            energy_per_atom = row.energy / len(atoms)
        
        # Calculate volume per atom
        volume_per_atom = atoms.get_volume() / len(atoms)
        
        # Extract forces if available
        try:
            forces = atoms.get_forces()
            force_norms = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_norms)
            # Append all forces for global distribution
            all_forces.extend(force_norms)
        except Exception:
            max_force = np.nan
        
        results.append({
            'energy_per_atom': energy_per_atom,
            'volume_per_atom': volume_per_atom,
            'natoms': len(atoms),
            'min_distance': min_dist,
            'max_force': max_force
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Verify we have data before plotting
    if df.empty:
        raise ValueError("No valid data was extracted from the database")
        
    # Check if we have the required columns
    required_columns = ['volume_per_atom', 'energy_per_atom', 'min_distance', 'natoms']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Set up the figure and axes
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 1. Energy/atom vs Volume/atom scatter
    valid_data = df.dropna(subset=['volume_per_atom', 'energy_per_atom'])
    if not valid_data.empty:
        sns.scatterplot(data=valid_data, x='volume_per_atom', y='energy_per_atom', ax=axs[0, 0])
        axs[0, 0].set_xlabel('Volume/Atom (Å³)')
        axs[0, 0].set_ylabel('Energy/Atom (eV)')
        axs[0, 0].set_title('Energy vs Volume per Atom')
    else:
        axs[0, 0].text(0.5, 0.5, 'No valid energy/volume data', ha='center', va='center')
        axs[0, 0].set_title('Energy vs Volume per Atom')
    
    # 2. Energy/atom distribution
    valid_energy = df['energy_per_atom'].dropna()
    if not valid_energy.empty:
        sns.histplot(valid_energy, kde=True, ax=axs[0, 1])
        axs[0, 1].set_xlabel('Energy/Atom (eV)')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].set_title('Energy per Atom Distribution')
    else:
        axs[0, 1].text(0.5, 0.5, 'No valid energy data', ha='center', va='center')
        axs[0, 1].set_title('Energy per Atom Distribution')
    
    # 3. Force distribution (all atoms)
    if all_forces:
        sns.histplot(all_forces, kde=True, ax=axs[0, 2])
        axs[0, 2].set_xlabel('Force Norm (eV/Å)')
        axs[0, 2].set_ylabel('Count')
        axs[0, 2].set_title('Force Distribution (All Atoms)')
    else:
        axs[0, 2].text(0.5, 0.5, 'No forces found', ha='center', va='center')
        axs[0, 2].set_title('Force Distribution (All Atoms)')
    
    # 4. Minimum distance distribution
    valid_dist = df['min_distance'].dropna()
    if not valid_dist.empty:
        sns.histplot(valid_dist, kde=True, ax=axs[1, 0])
        axs[1, 0].set_xlabel('Min. Distance (Å)')
        axs[1, 0].set_ylabel('Count')
        axs[1, 0].set_title('Minimum Distance Distribution')
    else:
        axs[1, 0].text(0.5, 0.5, 'No valid distance data', ha='center', va='center')
        axs[1, 0].set_title('Minimum Distance Distribution')
    
    # 5. Min distance vs Energy/atom scatter
    valid_dist_energy = df.dropna(subset=['min_distance', 'energy_per_atom'])
    if not valid_dist_energy.empty:
        sns.scatterplot(data=valid_dist_energy, x='min_distance', y='energy_per_atom', ax=axs[1, 1])
        axs[1, 1].set_xlabel('Min Distance (Å)')
        axs[1, 1].set_ylabel('Energy/Atom (eV)')
        axs[1, 1].set_title('Min Distance vs Energy per Atom')
    else:
        axs[1, 1].text(0.5, 0.5, 'No valid distance/energy data', ha='center', va='center')
        axs[1, 1].set_title('Min Distance vs Energy per Atom')
    
    # 6. Number of atoms distribution
    if not df['natoms'].empty:
        sns.histplot(df['natoms'], kde=True, ax=axs[1, 2])
        axs[1, 2].set_xlabel('Number of Atoms')
        axs[1, 2].set_ylabel('Count')
        axs[1, 2].set_title('System Size Distribution')
    else:
        axs[1, 2].text(0.5, 0.5, 'No atom count data', ha='center', va='center')
        axs[1, 2].set_title('System Size Distribution')
    
    # Save the figure if requested
    if output_dir:
        fig.savefig(f"{output_dir}/analysis.png", dpi=dpi, bbox_inches='tight')
    
    # Show or close the plot
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    return df
