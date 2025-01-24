# Import MPRester and initialize with your API key
from mp_api.client import MPRester
import sys

mpr = MPRester("KT7WcdIiIW97ZQ4E4SwkBiya7k2m1NBz")

# Specify the materials id
m_id = sys.argv[1]

# Get the structure object for that id
structure = mpr.get_structure_by_material_id(m_id)

# Write the POSCAR file to a file named "POSCAR"
structure.to(fmt="poscar", filename="POSCAR")
