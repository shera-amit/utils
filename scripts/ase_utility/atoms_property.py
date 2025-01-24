from ase.io import read
from ase.optimize import BFGS,FIRE,MDMin,QuasiNewton
from ase.constraints import ExpCellFilter,StrainFilter
from ase.spacegroup.symmetrize import FixSymmetry
import numpy as np
from scipy.optimize import minimize
import spglib

#Input: Input parameters
#Energy and Volume as np.array
def birch_murnaghan(input,volume,energy):
    """
    Returns the difference of a Birch-Murnaghan Fit

    Parameters
    ----------
    input : list
        List containing the parameters for the Birch-Murnaghan Equation of State (E0, V0, B0, B0dev)
    volume : numpy array
        Numpy array containing volumes from a energy-volume curve
    energy : numpy array
        Numpy array containing energies from a energy-volume curve

    Returns
    -------
    float
        Descrepancy between Birch-Murnaghan EOS and the energy-volume curve data
    """
    e0 = input[0]
    v0 = input[1]
    b0 = input[2]
    b0dev = input[3]
    part1 = b0dev*((v0/volume)**(2/3)-1)**3
    part2 = ((v0/volume)**(2/3)-1)**2*(6-4*(v0/volume)**(2/3))
    birchenergy = e0+9*v0*b0/16*(part1+part2)
    return np.sum((birchenergy-energy)**2)


def get_symmetry(atoms,symprec=1e-5,angle_tolerance=-1.0):
    cell = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    sym_obj = (cell,positions,numbers)
    return spglib.get_spacegroup(sym_obj, symprec=symprec,angle_tolerance=angle_tolerance)




class Optimizer:
    #self.atoms        initial atoms object
    def __init__(self,atoms):
        self.atoms = atoms
        self.constraints = False
        self.factor = 1


    ##################Input Functions##################
    #optimize_cell:   Shall the cell be optimized?
    #fmax:            Force tolerance value for finishing the minimization
    #constant_volume: If the cell is optimized, is the volume fixed? Important for energy volume curves
    #pressure:        Pressure, which will be applied in case of cell optimization, only applies when constant_volume=False
    #iterations:      How often over all different optimizers will be iterated.
    #steps:           Maximum steps for each optimizer
    #symmetry:        Whether symmetry will be fixed during minimization. 
    #symprec:         Accuracy parameter for symmetry determination
    def set_constraints(self,optimize_cell=False,fmax=1e-3,constant_volume=False,pressure=None,iterations=20,steps=100,symmetry=False,symprec=1e-4,mask=[1,1,1,1,1,1]):
        self.optimize_cell = optimize_cell
        if constant_volume==True and pressure!=None:
            raise Exception("Wrong Input","Cannot specify constant volume and pressure simulatenously!")
        if constant_volume==False and pressure==None and optimize_cell==True:
            raise Exception("Wrong Input","Pressure needs to be specified for not constant volume relaxation!")
        if self.optimize_cell==True:
            if pressure!=None:
                self.pressure = pressure/160.21766208
            else:
                self.pressure = pressure
            self.constant_volume = constant_volume
        self.fmax=fmax
        self.iterations=iterations
        self.steps=steps
        self.symmetry=symmetry
        self.symprec=symprec
        self.mask = mask
        self.constraints=True

    #Set the calculator
    def set_calculator(self,calculator):
        self.calculator = calculator


    def check_calculator(self):
        if not hasattr(self,"calculator"):
            raise Exception("No calculator attached","Please attach an ase calculator!")


    ##################Internal Functions##################
    #Calculates Stresses, Forces and Energies for initial configuration
    def __calc_static(self):
        atoms = self.atoms
        atoms.set_calculator(self.calculator)
        self.energy = atoms.get_potential_energy()
        self.forces = atoms.get_forces()
        self.stress = atoms.get_stress()


    #Runs the optimization based on the constraints
    def __run(self):
        if self.constraints==False:
            raise Exception("Not constraints set!","Please run set_constraints() first!")
        self.check_calculator()
        #Define DOFs for each minimization depending on whether 
        if self.optimize_cell:
            if self.constant_volume:
                dofs = ["p","a","p","a"]
            else:
                if self.pressure==0.0:
                    dofs = ["p","c","p","c","a"]
                else:
                    dofs = ["p","a","p","a"]       
        else:
            dofs = ["p"]
        
        #Order of the Optimizers
        optimizers = ["BFGS","LBFGS","FIRE","MDMin"]

        i = 0

        #Get start atoms object and get current forces
        self._current_atoms = self.atoms.copy()
        self._current_atoms.set_calculator(self.calculator)
        fcurrent = np.max(np.abs(self._current_atoms.get_forces()))

        #Define lists containing information about minimization
        self._opt_force_list = [fcurrent]
        self._opt_stress_list = [self._current_atoms.get_stress(voigt=False)]
        self._opt_energy_list = [self._current_atoms.get_potential_energy()]
        self._opt_atoms_list = [self._current_atoms.copy()]
        self._opt_optimizer_list = []
        self._opt_constraints_list = []

        #Stopping criterion, f<fmax or i exceeds the maximum number of iterations. If f is smaller than fmax in the first iteration 
        while fcurrent > self.fmax or (i<1):
            if i > self.iterations:
                print("Required Accuracy not reached! ("+str(fcurrent)+")")
                self.converged = False
                break
            for d in dofs:
                for o in optimizers:
                    self.__opt(d,o)
            back = len(dofs)*len(optimizers)
            eold = self._opt_energy_list[-1-back]
            enew = self._opt_energy_list[-1]
            if eold+1e-6 < enew and (not self.optimize_cell or self.constant_volume):
                raise Exception("Energy Error","Energy is increasing!")
            fcurrent = self._opt_force_list[-1]
            i += 1

        if not hasattr(self,"converged"):
            self.converged=True
        self.opt_atoms = self._opt_atoms_list[-1]
        self.opt_atoms.set_calculator(self.calculator)
        self.opt_energy = self.opt_atoms.get_potential_energy()
        self.opt_forces = self.opt_atoms.get_forces()
        self.opt_stress = self.opt_atoms.get_stress(voigt=False)
        self.opt_max_forces = np.max(np.abs(self.opt_atoms.get_forces()))


    #Optimizes the _current_atoms object and redefines it after optimization
    #Takes the DOF and ASE optimizer used
    def __opt(self,dof,ase_optimizer):
        atoms = self._current_atoms.copy()
        atoms.set_calculator(self.calculator)

        #Set symmetry constraint
        if self.symmetry:
            sym = FixSymmetry(atoms,symprec=self.symprec,verbose=False)
            atoms.set_constraint(sym)

        #Set cell constraints
        if dof=="p":
            in_atoms = atoms
        elif dof=="c":
            in_atoms = StrainFilter(atoms)
        elif dof=="a":
            if self.constant_volume==True:
                in_atoms = ExpCellFilter(atoms,mask=self.mask,constant_volume=True)
            else:
                if hasattr(self,"pressure"):
                    if self.pressure!=None:
                        in_atoms = ExpCellFilter(atoms,scalar_pressure=self.pressure,mask=self.mask)
                    else:
                        in_atoms = ExpCellFilter(atoms,mask=self.mask)
                else:
                    in_atoms = ExpCellFilter(atoms,mask=self.mask)
        else:
            raise Exception("Wrong Input","Degree of Freedom (dof) must be defined by 'a','p' or 'c'")


        #Choose ASE optimizer from input
        if ase_optimizer=="BFGS":
            opt = BFGS(in_atoms,alpha=70*self.factor,logfile="opt.txt")
        elif ase_optimizer=="FIRE":
            opt = FIRE(in_atoms,dt=0.1/self.factor,logfile="opt.txt")
        elif ase_optimizer=="LBFGS":
            opt = QuasiNewton(in_atoms,alpha=70*self.factor,logfile="opt.txt")
        elif ase_optimizer=="MDMin":
            opt = MDMin(in_atoms,dt=0.01/np.max(atoms.get_cell()[0:3])/self.factor,logfile="opt.txt")
        else:
            raise Exception("Wrong Input","Unknown optimizer"+str(ase_optimizer))
        
        #Run optimizer
        try:
            opt.run(fmax=self.fmax,steps=self.steps)
            new_atoms = atoms.copy()
        except RuntimeError:
            new_atoms = self._current_atoms.copy()
            self.factor = self.factor*2

        new_atoms.set_calculator(self.calculator)
        self._opt_force_list.append(np.max(np.abs(new_atoms.get_forces())))
        self._opt_stress_list.append(new_atoms.get_stress(voigt=False))
        self._opt_energy_list.append(new_atoms.get_potential_energy())
        self._opt_atoms_list.append(new_atoms)
        self._current_atoms = new_atoms


    ##################Output Methods##################
    #Energy of the minimized structure
    def get_minimized_potential_energy(self):
        if not hasattr(self,"opt_energy"):
            self.__run()
        return self.opt_energy

    #List of energies of all structures occured during minimization
    def get_steps_potential_energy(self):
        if not hasattr(self,"_opt_energy_list"):
            self.__run()
        return self._opt_energy_list
    
    #Energy of the initial structure
    def get_potential_energy(self):
        if not hasattr(self,"energy"):
            self.__calc_static()
        return self.energy
    

    #Same for Forces
    def get_minimized_forces(self):
        if not hasattr(self,"opt_forces"):
            self.__run()
        return self.opt_forces

    def get_steps_forces(self):
        if not hasattr(self,"_opt_force_list"):
            self.__run()
        return np.array(self._opt_force_list)

    def get_forces(self):
        if not hasattr(self,"forces"):
            self.__calc_static()
        return self.forces


    #Maximum force component in each iteration
    def get_steps_max_forces(self):
        if not hasattr(self,"forces"):
            self.__run()
        max_forces = []
        for i in range(0,len(self._opt_force_list)):
            max_forces.append(np.max(np.abs(self._opt_force_list[i])))
        return np.array(max_forces)


    #Same for Stresses (Conversion Factor for eV/A^3 in GPa)
    def get_minimized_stress(self):
        if not hasattr(self,"opt_stress"):
            self.__run()
        return self.opt_stress*160.21766208

    def get_steps_stress(self):
        if not hasattr(self,"_opt_stress_list"):
            self.__run()
        return np.array(self._opt_stress_list)*160.21766208

    def get_stress(self):
        if not hasattr(self,"stress"):
            self.__calc_static()
        return self.stress*160.21766208

    #Check whether calculation is converged
    def is_converged(self):
        if not hasattr(self,"converged"):
            self.__run()
        return self.converged


    def get_steps_atoms(self):
        if not hasattr(self,"_opt_atoms_list"):
            self.__run()
        return self._opt_atoms_list

    def get_minimized_atoms(self):
        if not hasattr(self,"opt_atoms"):
            self.__run()
        return self.opt_atoms




class EnergyVolumeCurve:
    #Class for the automatic creation of energy-volume curves
    #Takes an atoms object for initialisation 
    def __init__(self,atoms):
        self.atoms = atoms
        self.constraints = False


    #Settings:
    #Number of points at which a minimization is done
    #Maximum strain or maximum stress (in GPa), but not both 
    def set_constraints(self,points=11,max_strain=0.05,max_stress=None,fmax=1e-3,symmetry=False,symprec=1e-4):
        self.points=points
        if max_stress!=None and max_strain!=None:
            raise Exception("Too many parameters","Please specifiy only maximum stress OR strain!")
        if max_strain==None:
            self.strain_on = False
            self.max_stress= max_stress
        else:
            self.strain_on = True
            self.max_strain= max_strain
        self.symmetry = symmetry
        self.fmax = fmax
        self.symprec = symprec
        self.constraints = True
        
    #Set ASE calculator
    def set_calculator(self,calculator):
        self.calculator=calculator

    def check_calculator(self):
        if not hasattr(self,"calculator"):
            raise Exception("No calculator attached","Please attach an ase calculator!")

    #Run EV curve calculation
    def __run(self):
        self.check_calculator()
        if self.constraints==False:
            raise Exception("Not constraints set!","Please run set_constraints() first!")
        ref_atoms = self.atoms.copy()
        volumes = []
        energies = []
        stresses = []
        forces = []
        atoms_list = []
        for i in range(0,self.points):
            if self.strain_on:
                strain = (i-int(self.points/2))/(int(self.points/2))*self.max_strain
                new_cell = ref_atoms.get_cell()*(1+strain)**(1/3)
                strained_atoms = ref_atoms.copy()
                strained_atoms.set_cell(new_cell,scale_atoms=True)
                
                #Volume Optimization of this cell
                opt = Optimizer(strained_atoms)
                opt.set_calculator(self.calculator)
                opt.set_constraints(optimize_cell=True,
                                    fmax=self.fmax,
                                    constant_volume=True,
                                    pressure=None,
                                    symmetry=self.symmetry,
                                    symprec=self.symprec)

            else:
                stress = (i-int(self.points/2))/(int(self.points/2))*self.max_stress
                strained_atoms = ref_atoms.copy()

                #Stress Optimization of this cell
                opt = Optimizer(strained_atoms)
                opt.set_calculator(self.calculator)
                opt.set_constraints(optimize_cell=True,
                                    fmax=self.fmax,
                                    constant_volume=False,
                                    pressure=stress,
                                    symmetry=self.symmetry,
                                    symprec=self.symprec)

            atoms_list.append(opt.get_minimized_atoms())
            volumes.append(opt.get_minimized_atoms().get_volume())
            energies.append(opt.get_minimized_potential_energy())
            stresses.append(opt.get_minimized_stress())
            forces.append(opt.get_minimized_forces())
        self.volumes = np.array(volumes)
        self.energies = np.array(energies)
        self.forces = forces
        self.stresses = stresses
        self.atoms_list  = atoms_list


    def __run_birch_murnaghan(self):
        if not hasattr(self,"volumes"):
            self.__run()
        input_array = np.array([np.min(self.energies),self.volumes[np.argmin(self.energies)],50/160,1/160])
        min_result  = minimize(birch_murnaghan,input_array,args=(self.volumes,self.energies),method="BFGS")
        self.e0    = min_result.x[0]
        self.v0    = min_result.x[1]
        self.b0    = min_result.x[2]
        self.b0dev = min_result.x[3]


    def get_bulk_modulus(self):
        if not hasattr(self,"b0"):
            self.__run_birch_murnaghan()
        return self.b0*160.21766208

    def get_bulk_modulus_derivative(self):
        if not hasattr(self,"b0dev"):
            self.__run_birch_murnaghan()
        return self.b0dev

    def get_minimum_volume(self):
        if not hasattr(self,"v0"):
            self.__run_birch_murnaghan()
        return self.v0

    def get_minimum_energy(self):
        if not hasattr(self,"e0"):
            self.__run_birch_murnaghan()
        return self.e0


    def get_energy_list(self):
        if not hasattr(self,"energies"):
            self.__run()
        return self.energies

    def get_volume_list(self):
        if not hasattr(self,"volumes"):
            self.__run()
        return self.volumes

    def get_stress_list(self):
        if not hasattr(self,"stresses"):
            self.__run()
        return self.stresses

    def get_force_list(self):
        if not hasattr(self,"forces"):
            self.__run()
        return self.forces

    def get_atoms_list(self):
        if not hasattr(self,"atoms_list"):
            self.__run()
        return self.atoms_list

    def get_hydrostatic_pressures(self):
        if not hasattr(self,"stresses"):
            self.__run()
        hydro = []
        for i in range(0,self.points):
            hydro.append(np.mean(np.diag(self.stresses[i])))
        return np.array(hydro)

from ase import Atoms, Atom

class DimerCurve:

    def __init__(self,symbol1,symbol2):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        #self.energies = []
        #self.forces   = []
        #self.second_derivative = []
        self.constraints = False

    def set_constraints(self,min_dist,max_dist,step_size):
        self.min_dist  = min_dist
        self.max_dist  = max_dist
        self.step_size = step_size
        self.constraints = True


    def set_calculator(self,calculator):
        self.calculator=calculator

    def check_calculator(self):
        if not hasattr(self,"calculator"):
            raise Exception("No calculator attached","Please attach an ase calculator!")

    
    def __run(self):
        self.check_calculator()
        if self.constraints==False:
            raise Exception("Not constraints set!","Please run set_constraints() first!")
        distances = self.get_distances()
        energies = []
        forces   = []
        for d in distances:
            atoms = Atoms()
            atoms.set_cell([50,50,50])
            atoms.append(Atom(self.symbol1,[d/2,0,0]))
            atoms.append(Atom(self.symbol2,[-d/2,0,0]))
            atoms.set_calculator(self.calculator)
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces()[0][0])
        
        self.forces = forces
        self.energies = energies
        self.second_derivatives = np.gradient(self.forces,distances)

    def get_distances(self):
        return np.arange(self.min_dist,self.max_dist,self.step_size)

    def get_energy_list(self):
        if not hasattr(self,"energies"):
            self.__run()
        return self.energies

    def get_force_list(self):
        if not hasattr(self,"forces"):
            self.__run()
        return self.forces

    def get_second_derivatives_list(self):
        if not hasattr(self,"second_derivatives"):
            self.__run()
        return self.second_derivatives


        

    
    
                
                
            



    
    

        

    
