from ovito.io.ase import ase_to_ovito
from ase.io import write
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier, TimeAveragingModifier, SelectTypeModifier, DeleteSelectedModifier
from ovito.pipeline import StaticSource, Pipeline
import numpy as np
import os
import itertools
import form_factor_database
from ase.symbols import symbols2numbers
from scipy.constants import physical_constants
from ovito.data import CutoffNeighborFinder


def get_xray_form_factor(symbol, kdistances):
        para = form_factor_database.xray_form_factor[symbol]
        factors = np.zeros(len(kdistances))
        for i in range(0, 4):
            factors += para[2 * i] * np.exp(
                -para[2 * i + 1] * (kdistances / (np.pi * 4)) ** 2
            )
        return factors + para[-1]


#Works only for binary systems currently
#Use Debye formula, see https://static-content.springer.com/esm/art%3A10.1038%2Fnmat4447/MediaObjects/41563_2016_BFnmat4447_MOESM7_ESM.pdf

class XRDCalculator:
    def __init__(self, atoms) -> None:
        self.atoms = atoms
        self.cutoff = None
        self.spacing = None
        self.density = None
        self.num_density = None
        self.kmin = None
        self.kmax = None
        self.kspacing = None
        if isinstance(atoms, list):
            symbols = atoms[0].get_chemical_symbols()
        else:
            symbols = atoms.get_chemical_symbols()
        self.symbols = sorted(list(set(symbols)))
        self.pairs = list(itertools.combinations_with_replacement(self.symbols, 2))
        pass

    def set_constraints(self,mintheta=5, maxtheta=90, thetaspacing=0.2, cutoff=200, bins=20000, wavelength=1.5406):
        self.mintheta = mintheta
        self.maxtheta = maxtheta
        self.thetaspacing = thetaspacing
        self.cutoff = cutoff
        self.wavelength = wavelength
        self.thetas = np.arange(mintheta,maxtheta,thetaspacing)
        self.kdistances = 4*np.pi*np.sin(np.deg2rad(self.thetas))/self.wavelength
        self.bins= bins

    def _calculate_rdf(self):
        atoms  = self.atoms
        data = ase_to_ovito(atoms)
        pipeline = Pipeline(source=StaticSource(data=data))
        pipeline.modifiers.append(
           CoordinationAnalysisModifier(
                cutoff=self.cutoff,
                number_of_bins=self.bins,
                partial=True,
            )
        )
        data = pipeline.compute()

        distances = data.tables["coordination-rdf"].xy()[:, 0]

        self._calculate_density()
        num_density = self.num_density

        # Get Order of the Particle Types in Ovito and sort it
        ovito_sym = []
        for i in range(0, len(self.symbols)):
            ovito_sym.append(data.particles.particle_types.type_by_id(i + 1).name)
        ovito_pairs = list(itertools.combinations_with_replacement(ovito_sym, 2))

        bondCounts = list(np.zeros(len(self.pairs)))
        bondsDictRDF = {}
        for i in range(len(self.pairs)):
            if ovito_pairs[i] in self.pairs:
                index = self.pairs.index(ovito_pairs[i])
            else:
                index = self.pairs.index((ovito_pairs[i][1], ovito_pairs[i][0]))
            sym1 = self.pairs[index][0]
            sym2 = self.pairs[index][1]
            concentration1 = atoms.get_chemical_symbols().count(sym1)/len(atoms)
            concentration2 = atoms.get_chemical_symbols().count(sym2)/len(atoms)
            bondNum1 = data.tables["coordination-rdf"].xy()[:, i + 1]*4*np.pi*distances**2*num_density*concentration1*atoms.get_chemical_symbols().count(sym2)*(distances[1]-distances[0])
            bondNum2 = data.tables["coordination-rdf"].xy()[:, i + 1]*4*np.pi*distances**2*num_density*concentration2*atoms.get_chemical_symbols().count(sym1)*(distances[1]-distances[0])
            bondCounts[i]= bondNum1+bondNum2
            bondsDictRDF[sym1+","+sym2]= bondCounts[i]

        
        self.distances_rdf = distances
        self.bondsdict_rdf = bondsDictRDF
    


    def _calculate_density(self):
        if isinstance(self.atoms, list):
            volumes = []
            for atoms_objects in self.atoms:
                volumes.append(atoms_objects.get_volume())
            self.density = (
                1.660539e-24
                * np.sum(self.atoms[0].get_masses())
                / np.mean(np.array(volumes) * 1e-24)
            )  # In g/cm^3
            self.num_density = len(self.atoms[0]) / np.mean(volumes)
        else:
            self.density = (
                1.660539e-24
                * np.sum(self.atoms.get_masses())
                / (self.atoms.get_volume() * 1e-24)
            )
            self.num_density = len(self.atoms) / self.atoms.get_volume()
    
    def _calculate_distances(self):
        atoms  = self.atoms
        import matplotlib.pyplot as plt

        dataTotal = ase_to_ovito(atoms)
        
        finder = CutoffNeighborFinder(self.cutoff,dataTotal)



        histAll = np.zeros(self.bins)
        bin_edges = np.histogram([0],bins=self.bins,range=(0,self.cutoff),density=False)[1]
        self.distances = (bin_edges[1:]+bin_edges[:-1])/2

        print("COMPUTE NEIGHBORS TOTAL SYSTEM")
        for index in range(dataTotal.particles.count):
            histAll+=np.histogram(finder.neighbor_distances(index),bins=self.bins,range=(0,self.cutoff),density=False)[0]
        plt.plot(self.distances,histAll)
        plt.show()

        histdict = {}
        for symbols in self.symbols:
            pipeline = Pipeline(source=StaticSource(data=dataTotal))
            for sym2 in self.symbols:
                if sym2!=symbols:
                    modifier = SelectTypeModifier(property = 'Particle Type', types = {sym2})
                    pipeline.modifiers.append(modifier)
            pipeline.modifiers.append(DeleteSelectedModifier())
            data = pipeline.compute()
            finder = CutoffNeighborFinder(self.cutoff,data)
            histSub = np.zeros(self.bins)
            print("COMPUTE NEIGHBORS FOR "+str(symbols)+" SYSTEM")
            for index in range(data.particles.count):
                histSub+=np.histogram(finder.neighbor_distances(index),bins=self.bins,range=(0,self.cutoff),density=False)[0]
            plt.plot(self.distances,histSub)
            plt.show()
            histdict[symbols+","+symbols]=histSub

        bondsdict = histdict
        subtract = np.zeros(self.bins)
        for symbols in self.symbols:
            subtract += bondsdict[symbols+","+symbols]
            plt.plot(self.distances,bondsdict[symbols+","+symbols],label=symbols)
        bondsdict[self.symbols[0]+","+self.symbols[1]] = histAll-subtract
        plt.plot(self.distances,bondsdict[self.symbols[0]+","+self.symbols[1]],label="mixed")
        plt.legend()
        plt.show()

        self.bondsdict = bondsdict

    def _calculate_debye(self):
        kdistances = self.kdistances
        distances  = self.distances
        form_factor_dict = {}
        for symbols in self.symbols:
            form_factor_dict[symbols]=get_xray_form_factor(symbol=symbols,kdistances=kdistances)
        
        intensity = np.zeros(len(kdistances))

        for i in range(0,len(kdistances)):
            sum = 0
            for keys in self.bondsdict:
                  s1,s2 = keys.split(",")
                  for j in range(0,len(distances)):
                    sum += self.bondsdict[keys][j]*form_factor_dict[s1][i]*form_factor_dict[s2][i]*np.sin(kdistances[i]*distances[j])/(kdistances[i]*distances[j])
            intensity[i] = sum

        self.intensity = intensity


    def _calculate_debye_rdf(self):
        kdistances = self.kdistances
        distances  = self.distances_rdf
        form_factor_dict = {}
        for symbols in self.symbols:
            form_factor_dict[symbols]=get_xray_form_factor(symbol=symbols,kdistances=kdistances)
        
        intensity = np.zeros(len(kdistances))

        for i in range(0,len(kdistances)):
            sum = 0
            for keys in self.bondsdict_rdf:
                  s1,s2 = keys.split(",")
                  for j in range(0,len(distances)):
                    sum += self.bondsdict_rdf[keys][j]*form_factor_dict[s1][i]*form_factor_dict[s2][i]*np.sin(kdistances[i]*distances[j])/(kdistances[i]*distances[j])
            intensity[i] = sum

        self.intensity_rdf = intensity


    def get_xray_spectrum(self):
        self._calculate_distances()
        self._calculate_debye()
        return self.thetas, self.intensity
    
    def get_xray_spectrum_rdf(self):
        self._calculate_rdf()
        self._calculate_debye_rdf()
        return self.thetas, self.intensity_rdf





