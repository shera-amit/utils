from ase import Atoms,Atom
from atoms_property import Optimizer
import ase.build
import numpy as np

class Surface:

    def __init__(self,atoms):
        self.atoms = atoms

    def set_constraints(self,vacuum=5,cutoff=10,displacement=0.01,mesh=[100,100,100],t_step=10,t_min=0,t_max=1000):
        self.vacuum = vacuum
        self.cutoff  = cutoff

    def set_calculator(self,calculator):
        self.calculator = calculator


    def check_calculator(self):
        if not hasattr(self,"calculator"):
            raise Exception("No calculator attached","Please attach an ase calculator!")


    def __min_structure_calculation(self):
        opt = Optimizer(self.atoms)
        self.check_calculator()
        opt.set_calculator(self.calculator)
        opt.set_constraints(optimize_cell=True,fmax=1e-4,pressure=0)
        self.min_atoms = opt.get_minimized_atoms()
    

    def __construct_surfaces(self,hkl):
        if len(hkl)==3:
            h = hkl[0]
            k = hkl[1]
            l = hkl[2]
        if len(hkl)==4:
            u = hkl[0]
            v = hkl[1]
            s = hkl[2]
            w = hkl[3]
            a1,a2,a4 = self.atoms.cell
            a3 = -(a1+a2)
            sum = u*a1+v*a2+s*a3+w*a4
            h = np.dot(sum,a1)
            k = np.dot(sum,a2)
            l = np.dot(sum,a3)
            print(hkl)
        ase.build.surface(self.min_atoms,[h,k,l],5)