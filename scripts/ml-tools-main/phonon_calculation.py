from phonopy import Phonopy
from phonopy.units import VaspToTHz
from phonopy.structure.atoms import PhonopyAtoms
from ase import Atoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.phonon.band_structure import get_band_qpoints
from ase.dft.kpoints import get_special_points
import matplotlib.pyplot as plt
import numpy as np
from atoms_property import get_symmetry


class Phonon:
    
    def __init__(self,atoms):
        self.atoms = atoms
        self.run_mesh = False
        self.run_thermal = False
        self.run_force_constants = False
        self.phonon_dispersion = False
        self.phonon_dos = False
        self.phonon_pdos = False
        self.run_mesh_pdos = False


    def set_constraints(self,symprec=1e-5,cutoff=10,displacement=0.01,mesh="auto",t_step=10,t_min=0,t_max=1000,energy_convergence=1e-4):
        self.symprec = symprec
        self.cutoff  = cutoff
        self.displacement = displacement
        self.mesh = mesh #"auto"
        self.t_step = t_step
        self.t_min  = t_min
        self.t_max  = t_max
        self.energy_convergence = energy_convergence


    def set_calculator(self,calculator):
        self.calculator = calculator


    def check_calculator(self):
        if not hasattr(self,"calculator"):
            raise Exception("No calculator attached","Please attach an ase calculator!")


    def __force_constants_calculation(self):
        self.check_calculator()
        cell = self.atoms.cell.cellpar()
        self.atoms.set_pbc(True)
        x = int(np.ceil((2.1*self.cutoff)/cell[0]))
        y = int(np.ceil((2.1*self.cutoff)/cell[1]))
        z = int(np.ceil((2.1*self.cutoff)/cell[2]))
        #Convert ase atoms to Phonopy atoms
        self.phonon_atoms = PhonopyAtoms(symbols          = self.atoms.get_chemical_symbols(),
                                         cell             = self.atoms.get_cell(),
                                         scaled_positions = self.atoms.get_scaled_positions(wrap=True))
        self.phonon_object = Phonopy(self.phonon_atoms,[[x,0,0],[0,y,0],[0,0,z]],symprec=self.symprec,factor=VaspToTHz)
        self.phonon_object.generate_displacements(distance=self.displacement)
        self.supercells = self.phonon_object.get_supercells_with_displacements()

        force_list = []
        print("Number of Displacements: "+str(len(self.supercells)))
        self.spacegroup = get_symmetry(self.atoms,symprec=self.symprec)
        print("Space Group: "+str(self.spacegroup))
        if self.spacegroup!=get_symmetry(self.atoms,symprec=self.symprec*10):
            print("Warning: System shows a large sensitivity on the Symmetry Tolerance value. It might be advantageous in terms of computational time to increase the symmetry parameter, however, have a look that it does not become too large!")
        for i in range(0,len(self.supercells)):
            #Convert Phonopy atoms to ASE atoms
            ase_atoms = Atoms(symbols=list(self.supercells[i].get_chemical_symbols()),
                              positions=list(self.supercells[i].get_positions()),
                              cell=list(self.supercells[i].get_cell()),
                              pbc=True)
            ase_atoms.set_calculator(self.calculator)
            force_list.append(ase_atoms.get_forces())
        self.phonon_object.set_forces(force_list)
        self.phonon_object.produce_force_constants()
        self.run_force_constants=True
        
    def __run_mesh(self):
        if self.run_force_constants==False:
            self.__force_constants_calculation()
        if self.mesh=="auto":
            current_dens = 2
            self.phonon_object.run_mesh(current_dens)
            self.phonon_object.run_thermal_properties(t_step=self.t_step,t_max=self.t_max,t_min=self.t_min)
            fe = self.phonon_object.get_thermal_properties()[1]*0.01036/len(self.atoms)
            print(self.phonon_object.mesh_numbers)
            print("Iteration 0: ",current_dens,fe)
            dif = 1
            it = 1
            old_mesh = self.phonon_object.mesh_numbers
            while dif > self.energy_convergence:
                current_dens = current_dens*1.5
                self.phonon_object.run_mesh(current_dens)
                print(self.phonon_object.mesh_numbers)
                if np.linalg.norm(self.phonon_object.mesh_numbers-old_mesh)>1e-5:
                    self.phonon_object.run_thermal_properties(t_step=self.t_step,t_max=self.t_max,t_min=self.t_min)
                    fe_new = self.phonon_object.get_thermal_properties()[1]*0.01036
                    print("Iteration "+str(it)+": ",current_dens, np.max(np.abs(fe-fe_new)), fe_new)
                    dif = np.max(np.abs(fe-fe_new))
                    fe = fe_new
                    it = it+1
                    old_mesh = self.phonon_object.mesh_numbers
        else:
            self.phonon_object.run_mesh(self.mesh)
        print("Convergence finished")
        self.run_mesh = True

    def __run_mesh_pdos(self):
        if self.run_force_constants==False:
            self.__force_constants_calculation()
        self.phonon_object.run_mesh(self.mesh,with_eigenvectors=True, is_mesh_symmetry=False)
        self.run_mesh_pdos = True

    def __calculate_thermal_properties(self):
        if self.run_force_constants==False:
            self.__force_constants_calculation()
        if self.run_mesh==False:
            self.__run_mesh()
        self.phonon_object.run_thermal_properties(t_step=self.t_step,t_max=self.t_max,t_min=self.t_min)
        self.run_thermal = True

    def __run_phonon_dos(self):
        if self.run_force_constants==False:
            self.__force_constants_calculation()
        if self.run_mesh==False:
            self.__run_mesh()
        self.phonon_object.run_total_dos()
        self.phonon_dos = True

    def __run_phonon_pdos(self):
        if self.run_force_constants==False:
            self.__force_constants_calculation()
        if self.run_mesh_pdos==False:
            self.__run_mesh_pdos()
        self.phonon_object.run_projected_dos()
        self.phonon_pdos = True


    def get_auto_phonon_dispersion(self):
        if not self.run_mesh:
            self.__run_mesh()
        self.phonon_object.auto_band_structure(plot=True)
        self.phonon_dispersion=True

    #Input Points in the style GMXG, no coordinates are required
    def get_phonon_dispersion(self,points,exp_data=None):
        if self.run_force_constants == False:
            self.__force_constants_calculation()
        if self.run_mesh == False:
            self.__run_mesh()
        point_dict = get_special_points(self.atoms.cell,eps=self.symprec)
        coordinates= []
        high_sym_section = [0]
        for ch in points:
            coordinates.append(list(point_dict[ch]))

        total_path = 0
        qpoints, path = get_band_qpoints_and_path_connections([coordinates])

        self.phonon_object.run_band_structure(paths=qpoints)
        frequencies = self.phonon_object.get_band_structure_dict()["frequencies"]
        distances   = self.phonon_object.get_band_structure_dict()["distances"]
        x_list = []
        y_list = []

        frequencies = np.array(frequencies)
        for i in range(0,len(frequencies[0][0])):
            y_list.append([])
        for i in range(0,len(distances)):
            x_list.extend(list(distances[i]))
            for j in range(0,len(frequencies[0][0])):
                y_list[j].extend(list(frequencies[i,:,j]))
        self.dist_list_simple = x_list
        self.freq_list_simple = y_list
        self.phonon_dispersion=True

        for i in range(0,len(distances)):
            total_path += np.linalg.norm(np.array(distances[i][-1])-np.array(distances[i][0]))
            high_sym_section.append(total_path)

        self.high_sym_section = np.array(high_sym_section)/total_path

    
    def get_phonon_dispersion_frequencies(self):
        if not self.phonon_dispersion:
            self.get_auto_phonon_dispersion()
        return self.phonon_object.get_band_structure_dict()["frequencies"]

    def get_phonon_dispersion_distances(self):
        if not self.phonon_dispersion:
            self.get_auto_phonon_dispersion()
        return self.phonon_object.get_band_structure_dict()["distances"]

    def get_phonon_dispersion_distances_simple(self):
        if not self.phonon_dispersion:
            self.get_auto_phonon_dispersion()
        return self.dist_list_simple

    def get_phonon_dispersion_frequencies_simple(self):
        if not self.phonon_dispersion:
            self.get_auto_phonon_dispersion()
        return self.freq_list_simple

    def get_path_location_high_symmetry_points(self):
        if not self.phonon_dispersion:
            self.get_auto_phonon_dispersion()
        return self.high_sym_section

    def get_total_dos(self):
        if not self.phonon_dos:
            self.__run_phonon_dos()
        return self.phonon_object.get_total_DOS()
        
    def get_projected_dos(self):
        if not self.phonon_pdos:
            self.__run_phonon_pdos()
        return self.phonon_object.get_partial_DOS()

    def get_free_energy(self):
        if not self.run_thermal:
            self.__calculate_thermal_properties()
        temps, fe, entropy, cv = self.phonon_object.get_thermal_properties()
        return fe*0.01036 #Returns energy in eV and not kJ/mol, needs to be adjusted for other functions

    def get_entropy(self):
        if not self.run_thermal:
            self.__calculate_thermal_properties()
        temps, fe, entropy, cv = self.phonon_object.get_thermal_properties()
        return entropy

    def get_heat_capacity(self):
        if not self.run_thermal:
            self.__calculate_thermal_properties()
        temps, fe, entropy, cv = self.phonon_object.get_thermal_properties()
        return cv


    def get_temperatures(self):
        if not self.run_thermal:
            self.__calculate_thermal_properties()
        temps, fe, entropy, cv = self.phonon_object.get_thermal_properties()
        return temps

    def write_yaml(self):
        if not self.run_thermal:
            self.__calculate_thermal_properties()
        self.phonon_object.thermal_properties.write_yaml(filename="thermal_properties.yaml", volume=None)
        

        
            


    


    
