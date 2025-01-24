import os
from ml_io import write_ml,read_ml
import numpy as np
import matplotlib.pyplot as plt


def root_mean_square_error(a1,a2):
    return np.sqrt(np.sum((np.array(a1)-np.array(a2))**2)/len(a1))

#a1: Actual Value
#a2: Forecast Value
def mean_absolute_percentage_error(a1,a2):
    return np.sum(np.abs((np.array(a1)-np.array(a2))/np.array(a1)))/len(a1)

def mean_absolute_error(a1,a2):
    return np.sum(np.abs(np.array(a1)-np.array(a2)))/len(a1)

def maximum_absolute_difference(a1,a2):
    return np.max(np.abs(np.array(a1)-np.array(a2)))

def scatter_plot(a1,a2,case="energy",name="tag"):
    fig, axs = plt.subplots(2,sharex=True)
    axs[0].scatter(a1,a2)
    if case=="energy":
        axs[1].set_xlabel("DFT Energy (eV/atom)")
        axs[1].set_ylabel(r"$\Delta$E (eV/Atom)")
        axs[0].set_ylabel("MLP Energy (eV/atom)")
    elif case=="force":
        axs[1].set_xlabel(r"DFT Force (eV/$\AA$)")
        axs[1].set_ylabel(r"$\Delta$F (eV/$\AA$)")
        axs[0].set_ylabel(r"MLP Force (eV/$\AA$)")
    elif case=="stress":
        axs[1].set_xlabel(r"DFT Stress (GPa)")
        axs[1].set_ylabel(r"$\Delta \sigma$ (GPa)")
        axs[0].set_ylabel(r"MLP Stress (GPa)")
    max, min = get_max_min(a1,a2)
    axs[1].set_ylim(0,np.ceil(np.max(np.abs(np.array(a1)-np.array(a2)))))
    x = np.linspace(min,max,100)
    axs[0].plot(x,x)
    axs[1].scatter(a1,np.abs(np.array(a1)-np.array(a2)))
    plt.savefig(name+"."+case+".scatter.png")
    plt.close()



def calc_errors(e1,e2,f1,f2,s1,s2):
    out_s = "---------------------\nEnergy:\n"
    out_s = out_s+"RMSE:\t"+str(root_mean_square_error(e1,e2)*1000)+"\tmeV/atom\n"
    out_s = out_s+"MAE :\t"+str(mean_absolute_error(e1,e2)*1000)+"\tmeV/atom\n"
    out_s = out_s+"MAD :\t"+str(maximum_absolute_difference(e1,e2)*1000)+"\tmeV/atom\n"
    out_s = out_s+"MAPE:\t"+str(mean_absolute_percentage_error(e1,e2)*1000)+"\n"
    out_s = out_s+"---------------------\n\n---------------------\nForces:\n"
    out_s = out_s+"RMSE:\t"+str(root_mean_square_error(f1,f2))+"\teV/A\n"
    out_s = out_s+"MAE :\t"+str(mean_absolute_error(f1,f2))+"\teV/A\n"
    out_s = out_s+"MAD :\t"+str(maximum_absolute_difference(f1,f2))+"\teV/A\n"
    out_s = out_s+"MAPE:\t"+str(mean_absolute_percentage_error(f1,f2))+"\n"
    out_s = out_s+"---------------------\n\n---------------------\nStresses:\n"
    out_s = out_s+"RMSE:\t"+str(root_mean_square_error(s1,s2))+"\tGPa\n"
    out_s = out_s+"MAE :\t"+str(mean_absolute_error(s1,s2))+"\tGPa\n"
    out_s = out_s+"MAD :\t"+str(maximum_absolute_difference(s1,s2))+"\tGPa\n"
    out_s = out_s+"MAPE:\t"+str(mean_absolute_percentage_error(s1,s2))+"\n"
    out_s = out_s+"---------------------"
    return out_s

def get_max_min(a1,a2):
    max_value1 = np.max(a1)
    max_value2 = np.max(a2)
    max_value  = np.max([max_value1,max_value2])
    min_value1 = np.min(a1)
    min_value2 = np.min(a2)
    min_value  = np.min([min_value1,min_value2])
    return max_value,min_value


class Evaluator:
    def __init__(self,atoms_list):
        self.atoms_list = atoms_list
        self.energy_potential = []
        self.force_potential = []
        self.stress_potential = []
        self.energy_dft = []
        self.force_dft = []
        self.stress_dft = []
        self.config_types = []
        self.sub_config_type = []
        self.num_atoms = []
        self.calculated = False
        self.divided = False
        self.divided_smaller = False
        self.save_folder = "report"



    ##################Input Functions##################
    #tag:             Specify whether training or validation set
    #classifier:      Label in atoms.info, which specifies to which kind of subgroup an atoms object belongs to e.g. crystalline, amorphous
    #sub_classifier:  Label in atoms.info, which specifies to which kind of a subgroup of a subgroup an atoms object belong, e.g. polymorph for cystalline structures 
    def set_constraints(self,tag="training",classifier="config_type",sub_classifier=None,dft_energy_tag=None,dft_force_tag=None,dft_stress_tag=None,save_folder=None,energy_correction=None):
        self.classifier = classifier
        self.tag = tag
        self.sub_classifier = sub_classifier
        self.dft_energy_tag = dft_energy_tag
        self.dft_force_tag  = dft_force_tag
        self.dft_stress_tag = dft_stress_tag
        self.energy_correction = energy_correction
        if save_folder!=None:
            self.save_folder = save_folder
        os.system("mkdir "+self.save_folder)

    #Set the calculator
    def set_calculator(self,calculator,potentialname=None):
        self.calculator = calculator
        if potentialname!=None:
            self.potentialname = potentialname


    def check_calculator(self):
        if not hasattr(self,"calculator"):
            raise Exception("No calculator attached","Please attach an ase calculator!")

    def __calc_database(self):
        self.check_calculator()
        count =0 
        if self.calculator=="mlip":
            write_ml(".calc_in.cfg",self.atoms_list,format="mlip",specorder=["O","Si"],energy_name=None,force_name=None,virial_name=None)
            os.system("mpirun -np 10 mlp calc-efs "+self.potentialname+" .calc_in.cfg .calc_out.cfg")
            ref_atoms_list = read_ml(".calc_out.cfg","mlip")
        elif self.calculator=="runner":
            write_ml("input.data",self.atoms_list,format="nnp",energy_name=None,force_name=None,virial_name=None)
            os.system("rm output.data")
            os.system("RuNNer.x")
            ref_atoms_list =read_ml("output.data","nnp")
            stresses = np.loadtxt("nnstress.out",skiprows=1)
            for i in range(0,len(ref_atoms_list)):
                ref_atoms_list[i].info["stress"]=stresses[i*3:i*3+3,1:4]*27.211396/(0.529177208**3)
            

        for atoms in self.atoms_list:
            if self.energy_correction == None:
                correction = 0
            else:
                correction = 0
                for keys in self.energy_correction:
                    correction = correction - atoms.get_chemical_symbols().count(keys)*self.energy_correction[keys]

            if self.dft_energy_tag == None:
                self.energy_dft.append((atoms.get_potential_energy()+correction)/len(atoms))
            else:
                self.energy_dft.append((atoms.info[self.dft_energy_tag]+correction)/len(atoms))

            if self.dft_force_tag == None:
                self.force_dft.extend(list(atoms.get_forces().flatten()))
            else:
                self.force_dft.extend(list(atoms.arrays[self.dft_force_tag].flatten()))
            
            if self.dft_stress_tag == None:
                self.stress_dft.extend(list(atoms.get_stress(voigt=False).flatten()*160.21766208)) #Conversion to GPa
            else:
                self.stress_dft.extend(list(atoms.info[self.dft_stress_tag].flatten()*160.21766208)) #Conversion to GPa
                
            if self.calculator=="runner":
                self.energy_potential.append(ref_atoms_list[count].info["energy"]/len(atoms))
                self.force_potential.extend(list(ref_atoms_list[count].arrays["forces"].flatten()))
                self.stress_potential.extend(list(ref_atoms_list[count].info["stress"].flatten()*160.21766208))
            elif self.calculator=="mlip":
                self.energy_potential.append(ref_atoms_list[count].info["energy"]/len(atoms))
                self.force_potential.extend(list(ref_atoms_list[count].arrays["forces"].flatten()))
                self.stress_potential.extend(list(ref_atoms_list[count].info["stress"].flatten()*160.21766208))
            else:
                atoms.set_calculator(self.calculator)
                self.energy_potential.append(atoms.get_potential_energy()/len(atoms))
                self.force_potential.extend(list(atoms.get_forces().flatten()))
                self.stress_potential.extend(list(atoms.get_stress(voigt=False).flatten()*160.21766208)) #Conversion to GPa

            self.num_atoms.append(len(atoms))
            if self.classifier in atoms.info:
                self.config_types.append(atoms.info[self.classifier])
            else:
                self.config_types.append(None)
            if self.sub_classifier in atoms.info:
                self.sub_config_type.append(atoms.info[self.sub_classifier])
            else:
                self.sub_config_type.append(None)
            count+=1
        if self.calculator=="mlip":
            os.system("rm .calc_in.cfg .calc_out.cfg")
        self.calculated = True


    def __subdivide(self):
        self.type_list = []
        for types in self.config_types:
            if types == None:
                if not "other" in self.type_list:
                    self.type_list.append("other")
            else:
                if not types in self.type_list:
                    self.type_list.append(types)
        self.config_dict_force_dft= {}
        self.config_dict_force_pot= {}

        self.config_dict_energy_dft= {}
        self.config_dict_energy_pot= {}

        self.config_dict_stress_dft= {}
        self.config_dict_stress_pot= {}

        self.config_dict_nums = {}

        for types in self.type_list:
            energy_list_dft = []
            force_list_dft  = []
            stress_list_dft = []
            energy_list_potential = []
            force_list_potential  = []
            stress_list_potential = []
            config_num_list  = []
            for i in range(0,len(self.energy_dft)):
                if self.config_types[i]==types:
                    force_start = np.sum(self.num_atoms[0:i]*3)
                    force_end   = np.sum(self.num_atoms[0:i+1]*3)
                    energy_list_dft.append(self.energy_dft[i])
                    energy_list_potential.append(self.energy_potential[i])
                    force_list_dft.extend(self.force_dft[int(force_start):int(force_end)])
                    force_list_potential.extend(self.force_potential[int(force_start):int(force_end)])
                    stress_list_dft.extend(self.stress_dft[9*i:9*i+9])
                    stress_list_potential.extend(self.stress_potential[9*i:9*i+9])
                    config_num_list.append(self.num_atoms[i])
            self.config_dict_energy_dft[types]=energy_list_dft
            self.config_dict_energy_pot[types]=energy_list_potential
            self.config_dict_force_dft[types]=force_list_dft
            self.config_dict_force_pot[types]=force_list_potential
            self.config_dict_stress_dft[types]=stress_list_dft
            self.config_dict_stress_pot[types]=stress_list_potential
        self.divided=True


    def get_config_types(self):
        if self.calculated == False:
            self.__calc_database()
        if self.divided == False:
            self.__subdivide()
        return self.type_list    


    def print_errors(self):
        if self.calculated == False:
            self.__calc_database()
        outfile = open(self.save_folder+"/"+self.tag+".error.dat","w")
        s = calc_errors(self.energy_dft,self.energy_potential,self.force_dft,self.force_potential,self.stress_dft,self.stress_potential)
        outfile.write(s)
        outfile.close()

    def create_scatter_plot(self):
        if self.calculated == False:
            self.__calc_database()
        #Energies
        scatter_plot(self.energy_dft,self.energy_potential,case="energy",name=self.save_folder+"/"+self.tag)
        #Forces
        scatter_plot(self.force_dft,self.force_potential,case="force",name=self.save_folder+"/"+self.tag)
        #Stress
        scatter_plot(self.stress_dft,self.stress_potential,case="stress",name=self.save_folder+"/"+self.tag)


    def print_config_errors(self):
        if self.calculated == False:
            self.__calc_database()
        if self.divided == False:
            self.__subdivide()
        outfile = open(self.save_folder+"/"+self.tag+".config.error.dat","w")
        for types in self.type_list:
            s = calc_errors(self.config_dict_energy_dft[types],self.config_dict_energy_pot[types],
                            self.config_dict_force_dft[types],self.config_dict_force_pot[types],
                            self.config_dict_stress_dft[types],self.config_dict_stress_pot[types])
            outfile.write("\n"+types+":\n")
            outfile.write(s)
        outfile.close()

    def create_config_scatter_plots(self):
        if self.calculated == False:
            self.__calc_database()
        if self.divided == False:
            self.__subdivide()

        #Energy
        fig, axs = plt.subplots(2,sharex=True)
        max_list = []
        t_list = self.type_list.copy()
        for types in self.type_list:
            max_list.append(np.max(self.config_dict_energy_dft[types]))
            
        for i in range(0,len(self.type_list)):
            id = np.argmax(max_list)
            t  = t_list[id]
            del max_list[id]
            del t_list[id]
            axs[0].scatter(self.config_dict_energy_dft[types],self.config_dict_energy_pot[types],label=t)
            axs[1].scatter(self.config_dict_energy_dft[types],np.abs(np.array(self.config_dict_energy_dft[types])-np.array(self.config_dict_energy_pot[types])))
        max, min = get_max_min(self.energy_potential,self.energy_dft)
        axs[1].set_xlabel("DFT Energy (eV/atom)")
        axs[1].set_ylabel(r"$\Delta$E (eV/Atom)")
        axs[0].set_ylabel("MLP Energy (eV/atom)")
        plt.savefig(self.tag+".energy.config.scatter.png")
        plt.close()

        #Forces
        fig, axs = plt.subplots(2,sharex=True)
        max_list = []
        t_list = self.type_list.copy()
        for types in self.type_list:
            max_list.append(np.max(self.config_dict_force_dft[types])-np.min(self.config_dict_force_dft[types]))
            
        for i in range(0,len(self.type_list)):
            id = np.argmax(max_list)
            t  = t_list[id]
            del max_list[id]
            del t_list[id]
            axs[0].scatter(self.config_dict_force_dft[types],self.config_dict_force_pot[types],label=t)
            axs[1].scatter(self.config_dict_force_dft[types],np.abs(np.array(self.config_dict_force_dft[types])-np.array(self.config_dict_force_pot[types])))
        max, min = get_max_min(self.energy_potential,self.energy_dft)
        axs[1].set_xlabel("DFT Energy (eV/atom)")
        axs[1].set_ylabel(r"$\Delta$E (eV/Atom)")
        axs[0].set_ylabel("MLP Energy (eV/atom)")
        plt.savefig(self.tag+".force.config.scatter.png")
        plt.close()


        #Stress
        fig, axs = plt.subplots(2,sharex=True)
        max_list = []
        t_list = self.type_list.copy()
        for types in self.type_list:
            max_list.append(np.max(self.config_dict_stress_dft[types])-np.min(self.config_dict_stress_dft[types]))
            
        for i in range(0,len(self.type_list)):
            id = np.argmax(max_list)
            t  = t_list[id]
            del max_list[id]
            del t_list[id]
            axs[0].scatter(self.config_dict_stress_dft[types],self.config_dict_stress_pot[types],label=t)
            axs[1].scatter(self.config_dict_stress_dft[types],np.abs(np.array(self.config_dict_stress_dft[types])-np.array(self.config_dict_stress_pot[types])))
        max, min = get_max_min(self.energy_potential,self.energy_dft)
        axs[1].set_xlabel("DFT Energy (eV/atom)")
        axs[1].set_ylabel(r"$\Delta$E (eV/Atom)")
        axs[0].set_ylabel("MLP Energy (eV/atom)")
        plt.savefig(self.tag+".force.config.scatter.png")
        plt.close()



        

    
    def create_config_sub_scatter_plots(self):
        if self.calculated == False:
            self.__calc_database()
        if self.divided == False:
            self.__subdivide()
        for types in self.type_list:
            #Energy
            scatter_plot(self.config_dict_energy_dft[types],self.config_dict_energy_pot[types],case="energy",name=self.save_folder+"/"+self.tag+".config."+types)
            #Forces
            scatter_plot(self.config_dict_force_dft[types],self.config_dict_force_pot[types],case="force",name=self.save_folder+"/"+self.tag+".config."+types)
            #Stresses
            scatter_plot(self.config_dict_stress_dft[types],self.config_dict_stress_pot[types],case="stress",name=self.save_folder+"/"+self.tag+".config."+types)

