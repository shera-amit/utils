from ovito.io.ase import ase_to_ovito
from ovito.modifiers import CreateBondsModifier
from ovito.pipeline import StaticSource, Pipeline
import itertools
import numpy as np
from ovito.data import NearestNeighborFinder


def get_lowest_distances(atoms):
        data = ase_to_ovito(atoms)
        finder = NearestNeighborFinder(1,data)
        min_dist = np.inf
        for index in range(data.particles.count):
                for neigh in finder.find(index):
                        if min_dist > neigh.distance:
                                min_dist= neigh.distance
        return min_dist


def get_coordinations(atoms,cutoff=2):
        symbols = list(set(atoms.get_chemical_symbols()))
        data = ase_to_ovito(atoms)
        pipeline = Pipeline(source = StaticSource(data = data))
        
        mod = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)

        pairs = list(itertools.combinations_with_replacement(symbols,2))

        for p in pairs:
                if type(cutoff)==float or type(cutoff)==int:
                        mod.set_pairwise_cutoff(p[0],p[1],cutoff)
                else:
                        for keys in cutoff.keys():
                                elements = keys.split("-")
                                if elements[0]==p[0] and elements[1]==p[1]:
                                        mod.set_pairwise_cutoff(p[0],p[1],cutoff[keys])
                                elif elements[1]==p[0] and elements[0]==p[1]:
                                        mod.set_pairwise_cutoff(p[0],p[1],cutoff[keys])
                        

        
        
        pipeline.modifiers.append(mod)
        
        data = pipeline.compute()
        idx, count = np.unique(data.particles.bonds.topology, return_counts=True)

        bond_count = np.zeros(data.particles.count, dtype=int)
        bond_count[idx] = count
        
        return bond_count