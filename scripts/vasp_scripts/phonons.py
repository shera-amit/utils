import os
import shutil
import subprocess
import time
from ase.io import read
import numpy as np


class PhononCalculation:
    def __init__(self, poscar_file, run_script="run_phonon.py", submit_script="submit.sh"):
        self.poscar_file = poscar_file
        self.run_script = run_script
        self.submit_script = submit_script

        self.atoms = read(poscar_file, format="vasp")

    def pre_process(self, dim=(2, 2, 2)):
        self._pre_process(dim)

    def force_collect(self):
        self._force_collect()

    def clean_vasp_directory(self, path="."):
        self._clean_vasp_directory(path)

    def copy_contcar_to_poscar(self, path="."):
        self._copy_contcar_to_poscar(path)

    def _pre_process(self, dim):
        subprocess.run(["phonopy", "-d", "--dim=" + " ".join(map(str, dim))])

        files = os.listdir()
        poscar_files = [f for f in files if f.startswith('POSCAR-')]

        for poscar_file in poscar_files:
            dir_name = f"disp-{poscar_file.split('-')[1]}"

            os.makedirs(dir_name, exist_ok=True)
            shutil.copy(poscar_file, os.path.join(dir_name, "POSCAR"))

            shutil.copy(self.run_script, os.path.join(dir_name, "run_phonon.py"))
            shutil.copy(self.submit_script,
                        os.path.join(dir_name, "submit.sh"))

            subprocess.run(["sbatch", "submit.sh"], cwd=dir_name)

    def _force_collect(self):
        dirs = [d for d in os.listdir() if os.path.isdir(d)
                and d.startswith("disp-")]
        dirs.sort(key=lambda x: int(x.split("-")[1]))

        vasprun_files = [f"{d}/vasprun.xml" for d in dirs]
        subprocess.run(["phonopy", "-f"] + vasprun_files)
    
    def _clean_vasp_directory(self, path):
        files_to_keep = ["POSCAR", "CONTCAR"]

        for filename in os.listdir(path):
            if filename not in files_to_keep:
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def _copy_contcar_to_poscar(self, path):
        contcar_path = os.path.join(path, "CONTCAR")
        poscar_path = os.path.join(path, "POSCAR")
        if os.path.isfile(contcar_path):
            shutil.copy2(contcar_path, poscar_path)
        else:
            raise FileNotFoundError(f"CONTCAR file not found at {contcar_path}")
 


#TODO: implement post_process

    # def post_process(self, band_path, MP=(8, 8, 8), pdos=False):
    #     self._post_process(band_path, MP, pdos)
#TODO: implement post_process
    # def _post_process(self, band_path, MP, pdos):
    #     superatoms = read("SPOSCAR", format="vasp")
    #     elements = self._get_elements()
    #     replication = self._find_replication_factors(superatoms)

    #     mesh_params = {
    #         "ATOM_NAME": elements,
    #         "DIM": replication,
    #         "MP": MP
    #     }
    #     mesh_content = self.generate_config_content(mesh_params)
    #     self._create_config_file("mesh.conf", mesh_content)
    #     self._run_phonopy("-p", "mesh.conf", plot=True)
    #     self._run_phonopy("-t", "mesh.conf")
    #     self._run_phonopy("-t", "mesh.conf", plot=True)
    #     if pdos:
    #         pdos_params = {
    #             "ATOM_NAME": elements,
    #             "DIM": replication,
    #             "MP": MP,
    #             "PDOS": "1 2, 3 4 5 6"
    #         }
    #         pdos_content = self.generate_config_content(pdos_params)
    #         self._create_config_file("pdos.conf", pdos_content)
    #         self._run_phonopy("-p", "pdos.conf", plot=True)

    #     def format_label(label):
    #         return f"${label}$" if label == "GAMMA" else label

    #     band_labels = [" ".join(map(format_label, subpath))
    #                    for subpath in band_path['path']]
    #     band_labels_str = ",".join(band_labels)

    #     band_points = [band_path['kpoints'][label]
    #                    for subpath in band_path['path'] for label in subpath]
    #     separator_indices = [
    #         len(subpath) - 1 for subpath in band_path['path'][:-1]]

    #     band_str_parts = []
    #     for i, point in enumerate(band_points):
    #         band_str_parts.append(" ".join(map(str, point)))
    #         if i in separator_indices:
    #             band_str_parts.append(",")

    #     band_str = " ".join(band_str_parts)

    #     band_params = {
    #         "ATOM_NAME": elements,
    #         "DIM": replication,
    #         "BAND": band_str,
    #         "BAND_LABEL": band_labels_str
    #     }
    #     band_content = self.generate_config_content(band_params)
    #     self._create_config_file("band.conf", band_content)
    #     self._run_phonopy("-p", "band.conf", plot=True)

    # def _get_elements(self):
    #     elements = self.atoms.get_chemical_symbols()
    #     unique_elements = sorted(set(elements))
    #     return " ".join(unique_elements)

    # def _find_replication_factors(self, superatoms):
    #     unit_cell = self.atoms.get_cell()
    #     supercell = superatoms.get_cell()

    #     replication_factors = np.linalg.solve(
    #         unit_cell.T, supercell.T).round().astype(int)
    #     diagonal_elements = np.diagonal(replication_factors)
    #     return " ".join(str(x) for x in diagonal_elements)

    # def _create_config_file(self, filename, content):
    #     with open(filename, 'w') as f:
    #         f.write(content)

    # def _run_phonopy(self, command, config_file, plot=False, save_plot=False):
    #     args = ["phonopy", command, config_file]
    #     if plot:
    #         args.append("-p")
    #     if save_plot:
    #         args.append("-s")
    #     subprocess.run(args)

    # @staticmethod
    # def generate_config_content(params):
    #     content = []
    #     for key, value in params.items():
    #         if isinstance(value, (list, tuple)):
    #             value_str = " ".join(map(str, value))
    #         else:
    #             value_str = str(value)
    #         content.append(f"{key} = {value_str}")
    #     return "\n".join(content)
