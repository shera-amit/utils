from ase.io import read,write
from ml_io import read_ml,write_ml,read_runner_output
import os
import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes
import time
import subprocess
from ase.io import read,write

class MLIP(Calculator):
    implemented_properties = ["energy","forces","stress"]


    def __init__(self,potential_name="pot.mtp",specorder=None,label="mlip",directory="."):
        self.potential_name = potential_name
        self.specorder = specorder

        Calculator.__init__(self, label=label,directory=directory)

    def calculate(self,atoms=None, properties=None, system_changes=None):
        if properties is None:
            properties = self.implemented_properties
        if system_changes is None:
            system_changes = all_changes
            
        Calculator.calculate(self, atoms, properties, system_changes)
        self.run()




    def check_state(self, atoms, tol=1.0e-10):
        return Calculator.check_state(self, atoms, tol)



    def run(self,set_atoms=True):
        write_ml(self._directory+"/input.cfg",self.atoms,format="mlip",specorder=self.specorder,energy_name=None,force_name=None,virial_name=None,charge_name=None)
        subprocess.Popen("${ASE_MLIP_COMMAND} calc-efs "+self.potential_name+" "+self.directory+"/input.cfg "+self.directory+"/output.cfg",shell=True).wait()
        atoms = read_ml(self.directory+"/output.cfg","mlip",self.specorder)[0]
        if set_atoms:
            self.atoms = atoms
        self.results = {}
        self.results["energy"]= atoms.info["energy"]
        stress = atoms.info["stress"]
        self.results["stress"]= np.array([stress[0,0],stress[1,1],stress[2,2],stress[1,2],stress[0,2],stress[0,1]])
        self.results["forces"]= atoms.arrays["forces"]
        self.results["free_energy"] = atoms.info["energy"]


class RUNNER(Calculator):
    implemented_properties = ["energy","forces","stress"]


    def __init__(self,label="runner",directory="."):
        Calculator.__init__(self, label=label,directory=directory)

    def calculate(self,atoms=None, properties=None, system_changes=None):
        if properties is None:
            properties = self.implemented_properties
        if system_changes is None:
            system_changes = all_changes
            
        Calculator.calculate(self, atoms, properties, system_changes)
        self.run()




    def check_state(self, atoms, tol=1.0e-10):
        return Calculator.check_state(self, atoms, tol)



    def run(self,set_atoms=True):
        os.system("rm "+self._directory+"/input.data "+self.directory+"/nnatoms.out "+self.directory+"/nnforces.out "+self.directory+"/nnstress.out")
        write_ml(self._directory+"/input.data",self.atoms,format="nnp",energy_name=None,force_name=None,virial_name=None,charge_name=None)
        subprocess.Popen("${ASE_RUNNER_COMMAND} > pred.out",shell=True).wait()
        e,f,s = read_runner_output()
        atoms = read_ml(self._directory+"/input.data",format="nnp")[0]
        if set_atoms:
            self.atoms = atoms
        #os.system("rm "+self._directory+"/input.data")
        print(len(atoms))
        print(len(f))
        self.results = {}
        self.results["energy"]= e
        stress = s
        self.results["stress"]= np.array([stress[0,0],stress[1,1],stress[2,2],stress[1,2],stress[0,2],stress[0,1]])
        self.results["forces"]= f
        self.results["free_energy"] = e


