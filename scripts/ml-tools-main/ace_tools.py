from pyace.activelearning import read_extrapolation_data, compute_B_projections, compute_active_set, BBasisConfiguration
from ase.io import read, write
import pandas as pd
import numpy as np
from pyace import create_multispecies_basis_config
from pyace import PyACECalculator


def maxvol_selection(training_database: list, active_structures: list, potential_path: str):
    bconf = BBasisConfiguration(potential_path)

    elements = []

    for i in range(0,len(bconf.funcspecs_blocks)):
        name = bconf.funcspecs_blocks[i].block_name.split()
        for j in range(0,len(name)):
            if not name[j] in elements:
                elements.append(name[j])


    training_df = pd.DataFrame({"ase_atoms":training_database})
    active_df   = pd.DataFrame({"ase_atoms":active_structures})
    
    total_df    = pd.concat([training_df,active_df],axis=0)
    total_df.reset_index(drop=True,inplace=True)

    print("Compute Basis Functions")
    bproj = compute_B_projections(bconf,total_df["ase_atoms"],total_df.index)
    act_set = compute_active_set(*bproj,tol=1.00001,max_iters=1000)
    active_set_indices = act_set[1]

    #for i in range()
    #selection_indices = np.array(sorted(set(active_set_indices[0]) | set(active_set_indices[1])))

    selection_indices = []
    for i in range(0,len(elements)):
        selection_indices.extend(list(active_set_indices[i]))
    selection_indices = np.array(sorted(set(selection_indices)))

    selection_df = total_df.loc[selection_indices]
    new_selection_df = selection_df[selection_df.index>=len(training_df)]


    return new_selection_df["ase_atoms"].to_list()
    
    

def __create_basis(functions,rcut,elements):
    basis_configuration = create_multispecies_basis_config(
    {
      "deltaSplineBins": 0.001,
      "elements": elements,
      "embeddings":  {"ALL": {  "npot": 'FinnisSinclairShiftedScaled',
                                "fs_parameters": [ 1, 1],
                                "ndensity": 1 }
                     },
      "bonds":       {"ALL": {  "radbase": "SBessel",
                                "radparameters": [ 5.25 ],
                                "rcut": rcut,
                                "dcut": 0.01,
                                "NameOfCutoffFunction": "cos"}
                     },
      "functions":   { "number_of_functions_per_element": functions,
                       "UNARY":   { "nradmax_by_orders": [ 15, 6, 4, 3, 2, 2 ], "lmax_by_orders": [ 0 , 3, 3, 2, 2, 1 ]},
                       "BINARY":  { "nradmax_by_orders": [ 15, 6, 3, 2, 2, 1 ], "lmax_by_orders": [ 0 , 3, 2, 1, 1, 0 ]}
                     }

    
    })
    return basis_configuration

    
def get_basis_functions(atoms_list,rcut=5,functions_per_element=1000):
    elements = []
    for i in range(0,len(atoms_list)):
        elements = sorted(list(set(list(set(atoms_list[i].get_chemical_symbols()))+elements)))

    basis = __create_basis(functions_per_element,rcut,elements)
    calc  = PyACECalculator(basis)
    descriptors, ids = compute_B_projections(bconf=calc,atomic_env_list=atoms_list,structure_ind_list=range(0,len(atoms_list)))


    countList = []
    for i in range(0,len(elements)):
        countList.append(0)

    total_list = []
    for i in range(0,len(atoms_list)):
        symbols = atoms_list[i].get_chemical_symbols()
        des_list = []
        for j in range(0,len(atoms_list[i])):
            id = elements.index(symbols[j])
            des_list.append(descriptors[id][countList[id]])
            countList[id]+=1
        total_list.append(des_list)

    return total_list




