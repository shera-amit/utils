from ovito.io.ase import ase_to_ovito
from ase.io import write
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier, TimeAveragingModifier
from ovito.pipeline import StaticSource, Pipeline
import numpy as np
import os
import itertools
import form_factor_database
from ase.symbols import symbols2numbers
from scipy.constants import physical_constants


class StructureFactorCalculator:
    def __init__(self, atoms) -> None:
        self.atoms = atoms
        self.radial_distribution_functions = None
        self.cutoff = None
        self.spacing = None
        self.electron_structure_factor = None
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
        self._constraints = False
        self._calculated_partial_structure_factors = False
        self._calculated_radial_dist = False
        self._calculated_total_xray_structure_factor = False
        self._calculated_total_neutron_structure_factor = False
        self._calculated_total_electron_structure_factor = False
        self._calculated_xray_diffraction_intensity = False
        self._calculated_electron_diffraction_intensity = False
        self._calculated_neutron_diffraction_intensity = False
        self._calculated_xray_total_pdf = False
        self._calculated_electron_total_pdf = False
        self._calculated_neutron_total_pdf = False
        pass

    def set_constraints(
        self,
        cutoff=20,
        spacing=0.01,
        kspacing=0.05,
        kmin=0.5,
        kmax=25,
        xray_wavelength=None,
        neutron_wavelength=None,
        electron_wavelength=None,
    ):
        self.cutoff = cutoff
        self.spacing = spacing
        self.kspacing = kspacing
        self.kmax = kmax
        self.kmin = kmin
        self._constraints = True
        self.kdistances = np.arange(self.kmin, self.kmax, self.kspacing)
        self.xray_wavelength = xray_wavelength
        self.neutron_wavelength = neutron_wavelength
        self.electron_wavelength = electron_wavelength

    def _calc_single_atoms_radial_distribution_function(self, atoms_object):
        data = ase_to_ovito(atoms_object)
        pipeline = Pipeline(source=StaticSource(data=data))
        pipeline.modifiers.append(
            CoordinationAnalysisModifier(
                cutoff=self.cutoff,
                number_of_bins=int(self.cutoff / self.spacing),
                partial=True,
            )
        )
        data = pipeline.compute()
        distances = data.tables["coordination-rdf"].xy()[:, 0]

        # Get Order of the Particle Types in Ovito and sort it
        ovito_sym = []
        for i in range(0, len(self.symbols)):
            ovito_sym.append(data.particles.particle_types.type_by_id(i + 1).name)
        ovito_pairs = list(itertools.combinations_with_replacement(ovito_sym, 2))

        rdfs = list(np.zeros(len(self.pairs)))
        for i in range(len(self.pairs)):
            if ovito_pairs[i] in self.pairs:
                index = self.pairs.index(ovito_pairs[i])
            else:
                index = self.pairs.index((ovito_pairs[i][1], ovito_pairs[i][0]))
            rdfs[index] = data.tables["coordination-rdf"].xy()[:, i + 1]

        return distances, rdfs

    def _calculate_radial_distribution_functions(self):
        if isinstance(self.atoms, list):
            rdf_list = []
            count = 0
            for atoms_objects in self.atoms:
                print(
                    "CALCULATING RDF: "
                    + str(np.round(count / len(self.atoms), 2) * 100)
                    + " %"
                )
                distances, rdfs = self._calc_single_atoms_radial_distribution_function(
                    atoms_objects
                )
                rdf_list.append(rdfs)
                count += 1
            self.distances = distances
            self.radial_distribution_functions = np.mean(np.array(rdf_list), axis=0)
        else:
            distances, rdfs = self._calc_single_atoms_radial_distribution_function(
                self.atoms
            )
            self.distances = distances
            self.radial_distribution_functions = np.array(rdfs)
        self._calculated_radial_dist = True

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

    def _calculate_partial_structure_factors(self):
        sf_list = []
        rdfs = self.get_partial_radial_distribution_functions()
        for i in range(0, len(self.pairs)):
            sf_partial = (
                1
                + 4
                * np.pi
                * self.get_num_density()
                / self.kdistances
                * np.trapz(
                    np.sin(np.outer(self.kdistances, self.distances))
                    * (rdfs[i] - 1)
                    * self.distances,
                    x=self.distances,
                    axis=1,
                )
            )
            sf_list.append(sf_partial)
        self.partial_structure_factors = np.array(sf_list)
        self._calculated_partial_structure_factors = True

    def _calculate_total_structure_factor(self, type):
        if type == "xray":
            self._calc_xray_form_factors()
            form_factors = self.xray_form_factors
        elif type == "neutron":
            self._calc_neutron_form_factors()
            form_factors = self.neutron_form_factors
        elif type == "electron":
            self._calc_electron_form_factors()
            form_factors = self.electron_form_factors
        self._calc_concentrations()
        total_sf = np.zeros(len(self.get_partial_structure_factors()[0]))
        psf = self.get_partial_structure_factors()
        for i in range(0, len(self.pairs)):
            index1 = self.symbols.index(self.pairs[i][0])
            index2 = self.symbols.index(self.pairs[i][1])
            if index1 == index2:
                factor = 1
            else:
                factor = 2
            total_sf += (
                factor
                * self.concentrations[index1]
                * self.concentrations[index2]
                * form_factors[index1]
                * form_factors[index2]
                * psf[i]
            )

        normalization = 0
        for i in range(0, len(self.symbols)):
            normalization += self.concentrations[i] * form_factors[i]
        return total_sf / normalization**2

    def _calculate_diffraction_intensity(self, type, wavelength):
        if type == "xray":
            sf = self.get_total_xray_structure_factor()
            ff = self.xray_form_factors
        elif type == "neutron":
            sf = self.get_total_xray_structure_factor()
            ff = self.neutron_form_factors
        elif type == "electron":
            sf = self.get_total_electron_structure_factor()
            ff = self.electron_form_factors
        factor = 0
        for i in range(0,len(self.symbols)):
            factor += self.concentrations[i]*ff[i]

        max_k = 4*np.pi/wavelength
        max_index = 0
        for i in range(0,len(self.kdistances)):
            if self.kdistances[i] > max_k:
                max_index= i-1
                break

        return np.arcsin((wavelength*self.kdistances[:max_index])/(4*np.pi)), sf[:max_index]*factor[:max_index]**2

    def _calulate_total_xray_structure_factor(self):
        self.total_xray_structure_factor = self._calculate_total_structure_factor(
            "xray"
        )
        self._calculated_total_xray_structure_factor = True

    def _calculate_total_neutron_structure_factor(self):
        self.total_neutron_structure_factor = self._calculate_total_structure_factor(
            "neutron"
        )
        self._calculated_total_neutron_structure_factor = True

    def _calculate_total_electron_structure_factor(self):
        self.total_electron_structure_factor = self._calculate_total_structure_factor(
            "electron"
        )
        self._calculated_total_electron_structure_factor = True

    def _calculate_xray_diffraction_intensity(self):
        (
            self.xray_thetas,
            self.xray_diffraction_intensity,
        ) = self._calculate_diffraction_intensity("xray", self.xray_wavelength)
        self._calculated_xray_diffraction_intensity = True

    def _calculate_electron_diffraction_intensity(self):
        (
            self.electron_thetas,
            self.electron_diffraction_intensity,
        ) = self._calculate_diffraction_intensity("electron", self.electron_wavelength)
        self._calculated_electron_diffraction_intensity = True

    def _calculate_neutron_diffraction_intensity(self):
        (
            self.neutron_thetas,
            self.neutron_diffraction_intensity,
        ) = self._calculate_diffraction_intensity("neutron", self.neutron_wavelength)
        self._calculated_neutron_diffraction_intensity = True

    def _calculate_total_radial_distribution_function(self, type):
        if type == "xray":
            structure_factor = self.get_total_xray_structure_factor()
        elif type == "electron":
            structure_factor = self.get_total_electron_structure_factor()
        elif type == "neutron":
            structure_factor = self.get_total_neutron_structure_factor()
        ks = self.kdistances
        rs = self.distances

        G_reduced = (
            2
            / np.pi
            * np.trapz(
                np.sin(np.outer(self.distances, self.kdistances))
                * self.kdistances
                * (structure_factor - 1),
                x=self.kdistances,
                axis=1,
            )
        )
        g = G_reduced / (4 * np.pi * self.distances * self.num_density) + 1
        R = 4 * np.pi * self.distances**2 * self.num_density * g

        if type == "xray":
            self.total_xray_pdf = g
            self.total_xray_reduced_pdf = G_reduced
            self.total_xray_rdf = R
            self._calculated_xray_total_pdf = True
        elif type == "electron":
            self.total_electron_pdf = g
            self.total_electron_reduced_pdf = G_reduced
            self.total_electron_rdf = R
            self._calculated_electron_total_pdf = True
        elif type == "neutron":
            self.total_neutron_pdf = g
            self.total_neutron_reduced_pdf = G_reduced
            self.total_neutron_rdf = R
            self._calculated_neutron_total_pdf = True

    def _calc_concentrations(self):
        if isinstance(self.atoms, list):
            all_symbols = self.atoms[0].get_chemical_symbols()
        else:
            all_symbols = self.atoms.get_chemical_symbols()
        concentrations = []
        for sym in self.symbols:
            concentrations.append(all_symbols.count(sym) / len(all_symbols))
        self.concentrations = concentrations

    def _calc_xray_form_factors(self):
        form_factors = []
        for sym in self.symbols:
            form_factors.append(self.get_xray_form_factor(sym))
        self.xray_form_factors = form_factors

    def _calc_neutron_form_factors(self):
        form_factors = []
        for sym in self.symbols:
            form_factors.append(self.get_neutron_form_factor(sym))
        self.neutron_form_factors = form_factors

    def _calc_electron_form_factors(self):
        form_factors = []
        for sym in self.symbols:
            form_factors.append(self.get_electron_form_factor(sym))
        self.electron_form_factors = form_factors

    def check_constraints(self):
        if self._constraints == False:
            raise Exception(
                "Not constraints set!", "Please run set_constraints() first!"
            )

    def get_partial_radial_distribution_functions(self):
        self.check_constraints()
        if self._calculated_radial_dist == False:
            self._calculate_radial_distribution_functions()
        return self.radial_distribution_functions

    def get_distances(self):
        self.check_constraints()
        if self._calculated_radial_dist == False:
            self._calculate_radial_distribution_functions()
        return self.distances

    def get_kdistances(self):
        self.check_constraints()
        return self.kdistances

    def get_partial_structure_factors(self):
        self.check_constraints()
        if self._calculated_partial_structure_factors == False:
            self._calculate_partial_structure_factors()
        return self.partial_structure_factors

    def get_total_xray_structure_factor(self):
        self.check_constraints()
        if self._calculated_total_xray_structure_factor == False:
            self._calulate_total_xray_structure_factor()
        return self.total_xray_structure_factor

    def get_total_neutron_structure_factor(self):
        self.check_constraints()
        if self._calculated_total_neutron_structure_factor == False:
            self._calculate_total_neutron_structure_factor()
        return self.total_neutron_structure_factor

    def get_total_electron_structure_factor(self):
        self.check_constraints()
        if self._calculated_total_electron_structure_factor == False:
            self._calculate_total_electron_structure_factor()
        return self.total_electron_structure_factor

    def get_total_xray_radial_distribution_function(self):
        self.check_constraints()
        if self._calculated_xray_total_pdf == False:
            self._calculate_total_radial_distribution_function("xray")
        return self.total_xray_rdf

    def get_total_xray_pair_distribution_function(self):
        self.check_constraints()
        if self._calculated_xray_total_pdf == False:
            self._calculate_total_radial_distribution_function("xray")
        return self.total_xray_pdf

    def get_total_xray_reduced_pair_distribution_function(self):
        self.check_constraints()
        if self._calculated_xray_total_pdf == False:
            self._calculate_total_radial_distribution_function("xray")
        return self.total_xray_reduced_pdf

    def get_total_electron_radial_distribution_function(self):
        self.check_constraints()
        if self._calculated_electron_total_pdf == False:
            self._calculate_total_radial_distribution_function("electron")
        return self.total_electron_rdf

    def get_total_electron_pair_distribution_function(self):
        self.check_constraints()
        if self._calculated_electron_total_pdf == False:
            self._calculate_total_radial_distribution_function("electron")
        return self.total_electron_pdf

    def get_total_electron_reduced_pair_distribution_function(self):
        self.check_constraints()
        if self._calculated_electron_total_pdf == False:
            self._calculate_total_radial_distribution_function("electron")
        return self.total_electron_reduced_pdf

    def get_total_neutron_radial_distribution_function(self):
        self.check_constraints()
        if self._calculated_neutron_total_pdf == False:
            self._calculate_total_radial_distribution_function("neutron")
        return self.total_neutron_rdf

    def get_total_neutron_pair_distribution_function(self):
        self.check_constraints()
        if self._calculated_neutron_total_pdf == False:
            self._calculate_total_radial_distribution_function("neutron")
        return self.total_neutron_pdf

    def get_total_neutron_reduced_pair_distribution_function(self):
        self.check_constraints()
        if self._calculated_neutron_total_pdf == False:
            self._calculate_total_radial_distribution_function("neutron")
        return self.total_neutron_reduced_pdf

    def get_xray_diffraction_intensity(self):
        self.check_constraints()
        if self._calculated_xray_diffraction_intensity == False:
            self._calculate_xray_diffraction_intensity()
        return self.xray_diffraction_intensity

    def get_xray_thetas(self):
        self.check_constraints()
        if self._calculated_xray_diffraction_intensity == False:
            self._calculate_xray_diffraction_intensity()
        return self.xray_thetas

    def get_electron_diffraction_intensity(self):
        self.check_constraints()
        if self._calculated_electron_diffraction_intensity == False:
            self._calculate_electron_diffraction_intensity()
        return self.electron_diffraction_intensity

    def get_electron_thetas(self):
        self.check_constraints()
        if self._calculated_electron_diffraction_intensity == False:
            self._calculate_electron_diffraction_intensity()
        return self.electron_thetas

    def get_neutron_diffraction_intensity(self):
        self.check_constraints()
        if self._calculated_neutron_diffraction_intensity == False:
            self._calculate_neutron_diffraction_intensity()
        return self.neutron_diffraction_intensity

    def get_neutron_thetas(self):
        self.check_constraints()
        if self._calculated_neutron_diffraction_intensity == False:
            self._calculate_neutron_diffraction_intensity()
        return self.neutron_thetas

    def get_num_density(self):
        if self.num_density == None:
            self._calculate_density()
        return self.num_density

    def get_density(self):
        if self.density == None:
            self._calculate_density()
        return self.density

    """
    Brown P J, Fox A G, Maslen E N, O'Keefe M A and Willis B T M 2004 Intensity
    of diffraction intensities International Tables for Crystallography
    Volume C: Mathematical, Physical, and Chemical Tables
    Table 6.1.1.4
    """

    def get_xray_form_factor(self, symbol):
        para = form_factor_database.xray_form_factor[symbol]
        factors = np.zeros(len(self.kdistances))
        for i in range(0, 4):
            factors += para[2 * i] * np.exp(
                -para[2 * i + 1] * (self.kdistances / (np.pi * 4)) ** 2
            )
        return factors + para[-1]

    def get_neutron_form_factor(self, symbol):
        para = form_factor_database.neutron_form_factor[symbol]
        return para

    def get_electron_form_factor(self, symbol):
        xray_form_factor = self.get_xray_form_factor(symbol)
        Z = symbols2numbers(symbol)
        prefactor = 1 / (
            8 * np.pi**2 * physical_constants["Bohr radius"][0] * 1e10
        )  # Convert into 1/A
        return prefactor * ((Z - xray_form_factor) / self.kdistances**2)
