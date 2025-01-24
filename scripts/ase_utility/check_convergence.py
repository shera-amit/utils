from pathlib import Path
from ase.calculators.vasp import Vasp
import sys

def check_convergence(glob_pattern):
    for p in Path('.').glob(glob_pattern):
        directory = p.parent
        calc = Vasp(directory=directory)
        
        if calc.read_convergence():
            print(directory, 'converged')
        else:
            print(directory, 'did not converge')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_convergence.py <glob_pattern>")
        sys.exit(1)
    
    glob_pattern = sys.argv[1]
    check_convergence(glob_pattern)

