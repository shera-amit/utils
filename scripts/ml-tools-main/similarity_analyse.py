from dscribe.descriptors import SOAP
import numpy as np
from coordination_calculator import get_lowest_distances

def sim(p1,p2):
	return (np.dot(p1,p2)/(np.sqrt(np.dot(p1,p1)*np.dot(p2,p2))))**4



def get_similarities(atoms,atoms_ref,atom_index=0,rcut=5,nmax=8,lmax=8,sigma=0.5,scale=False):
    symbols = atoms.get_chemical_symbols()

    if scale==False:
        soap    = SOAP(species=list(set(symbols)),periodic=True,r_cut=rcut,n_max=nmax,l_max=lmax,sigma=sigma)
        ref_symbol = atoms_ref.get_chemical_symbols()[atom_index]
        ref_vector = soap.create(atoms_ref)[atom_index]
        atoms_vector = soap.create(atoms)
    else:
        distAtoms = get_lowest_distances(atoms)
        distRef   = get_lowest_distances(atoms_ref)

        avgDist = (distAtoms+distRef)/2

        mod_cutoff = rcut/avgDist
        new_sigma  = sigma/avgDist

        new_atoms = atoms.copy()
        new_atoms.set_cell(atoms.get_cell()/distAtoms,scale_atoms=True)

        new_atoms_ref = atoms_ref.copy()
        new_atoms_ref.set_cell(atoms_ref.get_cell()/distRef,scale_atoms=True)

        soap    = SOAP(species=list(set(symbols)),periodic=True,r_cut=mod_cutoff,n_max=nmax,l_max=lmax,sigma=new_sigma)

        ref_symbol = new_atoms_ref.get_chemical_symbols()[atom_index]
        ref_vector = soap.create(new_atoms_ref)[atom_index]
        atoms_vector = soap.create(new_atoms)


    output_list = []
    for i in range(0,len(atoms)):
        if symbols[i]==ref_symbol:
            output_list.append(sim(atoms_vector[i],ref_vector))


    return np.array(output_list)


def get_similarities_structures(atoms1,atoms2,scale=False,average="inner",rcut=5,nmax=8,lmax=8,sigma=0.5):
    symbols = atoms1.get_chemical_symbols()

    if scale==False:
        soap    = SOAP(species=list(set(symbols)),periodic=True,r_cut=rcut,n_max=nmax,l_max=lmax,sigma=sigma,average=average)
        atoms1_vector = soap.create(atoms1)
        atoms2_vector = soap.create(atoms2)
    else:
        distAtoms1 = get_lowest_distances(atoms1)
        distAtoms2 = get_lowest_distances(atoms2)

        avgDist = (distAtoms1+distAtoms2)/2

        mod_cutoff = rcut/avgDist
        new_sigma  = sigma/avgDist

        new_atoms1 = atoms1.copy()
        new_atoms1.set_cell(atoms1.get_cell()/distAtoms1,scale_atoms=True)

        new_atoms2 = atoms2.copy()
        new_atoms2.set_cell(atoms2.get_cell()/distAtoms2,scale_atoms=True)

        soap = SOAP(species=list(set(symbols)),periodic=True,r_cut=mod_cutoff,n_max=nmax,l_max=lmax,sigma=new_sigma,average=average)
        atoms1_vector = soap.create(new_atoms1)
        atoms2_vector = soap.create(new_atoms2)



    return sim(atoms1_vector,atoms2_vector) 
        
