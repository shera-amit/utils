from dataclasses import replace
from operator import mod
import numpy as np
from dscribe.descriptors import SOAP
from sklearn import linear_model
import matplotlib.pyplot as plt


def calc_rmse(a1,a2):
	a1 = np.array(a1)
	a2 = np.array(a2)
	return np.sqrt(np.sum((a1-a2)**2)/len(a1))


class PropertyFit:

    def __init__(self,atoms_list):
        self.atoms_list = atoms_list


    #species: Species for which the NMR shifts shall be predicted
    #cutoff: Cutoff radius of the descriptor
    #nmax,lmax: Expansion coefficients for the SOAP
    #sigma: Width of the Gaussians
    #Crossover: How the soap is determined for different elements, makes no difference in the result I think
    #Periodic: Are the structures periodic?
    #Min_shift, max_shift: Set the training and validation shifts to a maximum and minimum value 
    #validation_set: Size of the validation set
    def set_constraints(self,species,nmr_label="nmr_shifts",cutoff=5,nmax=5,lmax=5,sigma=0.3,crossover=True,periodic=True,min_shift=None,max_shift=None,validation_set=0.1):
        elements = []
        for atoms in self.atoms_list:
            for s in atoms.get_chemical_symbols():
                if not s in elements:
                    elements.append(s)
        self.species    = species
        self.des_para = [elements,periodic,cutoff,nmax,lmax,sigma,crossover]
        self.descriptor = SOAP(species=elements,periodic=periodic,rcut=cutoff,nmax=nmax,lmax=lmax,sigma=sigma,crossover=crossover)
        self.nmr_label  = nmr_label
        self.min_shift  = min_shift
        self.max_shift  = max_shift
        self.validation_set = validation_set

    def __prepare(self):
        nmr_list  = []
        soap_list = []
        for atoms in self.atoms_list:
            soap_vec = self.descriptor.create(atoms)
            nmr_vec  = atoms.arrays[self.nmr_label]
            symbols  = atoms.get_chemical_symbols()

            for i in range(0,len(atoms)):
                if symbols[i]==self.species:
                    nmr_list.append(nmr_vec[i])
                    soap_list.append(soap_vec[i])
        self.soap_list = soap_list
        self.nmr_list  = nmr_list
    
    def __split(self):
        reduced_nmr  = []
        reduced_soap = []

        if self.max_shift != None:
            max = self.max_shift
        else:
            max = np.max(self.nmr_list)

        if self.min_shift != None:
            min = self.min_shift
        else:
            min = np.min(self.nmr_list)
        

        for i in range(0,len(self.nmr_list)):
            if min < self.nmr_list[i] and max > self.nmr_list[i]:
                reduced_nmr.append(self.nmr_list[i])
                reduced_soap.append(self.soap_list[i])
        
        num_valid = int(len(reduced_nmr)*self.validation_set)
        
        valid_nmr  = []
        valid_soap = []

        training_nmr  = []
        training_soap = []

        valid_ids = np.random.choice(len(reduced_nmr),size=num_valid,replace=False)
        
        for i in range(0,len(reduced_nmr)):
            if i in valid_ids:
                valid_nmr.append(reduced_nmr[i])
                valid_soap.append(reduced_soap[i])
            else:
                training_nmr.append(reduced_nmr[i])
                training_soap.append(reduced_soap[i])
        
        self.training_nmr    = training_nmr
        self.training_soap   = training_soap
        self.validation_nmr  = valid_nmr
        self.validation_soap = valid_soap

    def fit(self,alpha=0.01):
        self.__prepare()
        self.__split()
        X = np.array(self.training_soap)
        Y = np.array(self.training_nmr)
        reg = linear_model.Ridge(alpha=alpha)
        reg.fit(X,Y)
        self.model = reg
        self.predictor = PropertyPredict(species=self.species,descriptor=self.descriptor,model=reg,parameters=self.des_para)

    def get_predictor(self):
        return self.predictor


    #returns training, test error and full error (including outliners)
    def get_errors(self):
        X = np.array(self.training_soap)
        Y = np.array(self.training_nmr)
        X_valid = np.array(self.validation_soap)
        Y_valid = np.array(self.validation_nmr)
        X_full = np.array(self.soap_list)
        Y_full = np.array(self.nmr_list)


        Y_pred       = self.model.predict(X)
        Y_pred_valid = self.model.predict(X_valid)
        Y_pred_full  = self.model.predict(X_full)

        return calc_rmse(Y,Y_pred),calc_rmse(Y_valid,Y_pred_valid),calc_rmse(Y_full,Y_pred_full)
        


    def plot_scatter_plot(self,name="scatter_plot.png",xlabel="DFT NMR Shift",ylabel="LRR NMR Shift",full=False):
        rmse_train, rmse_valid, rmse_full = self.get_errors()
        if full==False:
            X = np.array(self.training_soap)
            Y = np.array(self.training_nmr)
            X_valid = np.array(self.validation_soap)
            Y_valid = np.array(self.validation_nmr)

            Y_pred = self.model.predict(X)
            Y_pred_valid = self.model.predict(X_valid)
            plt.scatter(Y,Y_pred,label="Training (RMSE: "+str(np.round(rmse_train,2))+")")
            plt.scatter(Y_valid,Y_pred_valid,label="Validation (RMSE: "+str(np.round(rmse_valid,2))+")")
            min  = np.min(np.concatenate((Y,Y_pred,Y_valid,Y_pred_valid)))
            max  = np.max(np.concatenate((Y,Y_pred,Y_valid,Y_pred_valid)))
            plt.plot(np.linspace(min,max,num=100),np.linspace(min,max,num=100),color="black")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.savefig(name)
            plt.clf()
        else:
            X = np.array(self.soap_list)
            Y = np.array(self.nmr_list)

            Y_pred = self.model.predict(X)
            plt.scatter(Y,Y_pred,label="Full (RMSE: "+str(np.round(rmse_full))+")")
            min  = np.min(np.concatenate((Y,Y_pred)))
            max  = np.max(np.concatenate((Y,Y_pred)))
            plt.plot(np.linspace(min,max,num=100),np.linspace(min,max,num=100),color="black")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.savefig(name)
            plt.clf()



        


import joblib


class PropertyPredict:

    def __init__(self,path=None,species=None,descriptor=None,model=None,parameters=None):
        if species==None and descriptor==None and model==None and parameters==None and path==None:
            raise Exception("Please add arguments for initialization")
        else:
            if (species!=None or descriptor!=None or model!=None or parameters!=None) and path!=None:
                raise Exception("Please only specifiy path OR parameters for initalization")
            else:
                if path==None:
                    self.__init_model(species,descriptor,model,parameters)
                else:
                    self.__load_model(path)
        

    def __init_model(self,species,descriptor,model,parameters):
        self.species = species
        self.descriptor = descriptor
        self.model = model
        self.parameters = parameters

    def __load_model(self,name):
        self.model = joblib.load(name)
        infile = open(name+".para","r")
        data   = infile.readlines()
        self.species = str(data[0][:-1])
        elements = data[1].split(",")[:-1]
        periodic = bool(int(data[2]))
        cutoff   = float(data[3])
        nmax     = int(data[4])
        lmax     = int(data[5])
        sigma    = float(data[6])
        crossover= bool(int(data[7]))
        self.descriptor = SOAP(species=elements,periodic=periodic,rcut=cutoff,nmax=nmax,lmax=lmax,sigma=sigma,crossover=crossover)
        self.parameters = [elements,periodic,cutoff,nmax,lmax,sigma,crossover]

        
    def save_model(self,name):
        joblib.dump(self.model,name)
        outfile = open(name+".para","w")
        outfile.write(self.species+"\n")  #species
        for e in self.parameters[0]:      #elements
            outfile.write(e+",")
        outfile.write("\n")
        outfile.write(str(int(self.parameters[1]))+"\n") #periodic
        outfile.write(str(self.parameters[2])+"\n")      #cutoff
        outfile.write(str(self.parameters[3])+"\n")      #nmax
        outfile.write(str(self.parameters[4])+"\n")      #lmax
        outfile.write(str(self.parameters[5])+"\n")      #sigma
        outfile.write(str(int(self.parameters[6]))+"\n")    #crossover
        outfile.close()


    #Returns a list of atoms properties, but only for the specific type and corresponding IDs
    def predict(self,atoms):
        soap_vec = self.descriptor.create(atoms)
        des_list = []
        id_list  = []
        for i in range(0,len(atoms)):
            if atoms.get_chemical_symbols()[i]==self.species:
                des_list.append(soap_vec[i])
                id_list.append(i)
        X = np.array(des_list)
        Y = self.model.predict(X)
        return Y,np.array(id_list)
        


        