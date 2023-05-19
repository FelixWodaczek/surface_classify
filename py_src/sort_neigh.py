import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import os
import pathlib
import pickle
import sys
from time import time

from ase.atoms import Atoms
from ase.io import write as ase_write
from ase.io import read as ase_read
from ase.build import sort as ase_sort
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.cell import Cell

from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel as ds_AverageKernel

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

STANDARD_LOCAL_STRUCTURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),"src/localstructures/")

class BaseClassifier():
    def __init__(self, first_class=3, last_class=9, non_class_max=14, **kwargs):
        """Base classifier, load generic classification parameters, define basic functions.

        Args:
            first_class (int, optional): Minimum number of neighbours to classify. Defaults to 3.
            last_class (int, optional): Maximum number of neighbours to classify. Defaults to 9.
            non_class_max (int, optional): Maximum number of neighbours to consider. Defaults to 14.
        """
        self.class_low = first_class - 1
        self.class_high = last_class + 1
        self.low_list = [ii for ii in range(0, self.class_low + 1)]
        self.high_list = [ii for ii in range(self.class_high, non_class_max+1)]
        self.non_class_list = self.low_list + self.high_list

    def load_identifiers(self, **kwargs):
        """Base function for loading identifiers into classification dict.
        """
        pass

    def classify(self, **kwargs):
        """Base classification function.
        """
        pass

class NeighbourClassifier(BaseClassifier):
    def __init__(self, local_structures_path=None, descr_species=["Rh", "Cu"], **kwargs):
        """Classifier for determining the type of configuration according to values found in `self.identification_dict`.
        `self.identification_dict` needs to be loaded from `local_structures_path` using `self.load_identifiers`, where local_structures path should have the following structure:
        local_structures_path/
            identifiers.json # names for each structure
            a.extxyz # structure file a
            b.extxyz # structure file b
            c.extxyz # structure file c

        see the /src/ folder in this repository for usable examples of such files.

        Args:
            local_structures_path (path, optional): Path of folder containing local structures for identification dictionary. Defaults to None.
            descr_species (list, optional): Species to build soap descriptors for. Defaults to ["Rh", "Cu"].
        """
        super().__init__(**kwargs)

        self.local_structures_path = None
        self.info_dict = {}
        self.identification_list = []
        self.identification_dict = {}
        self.n_classes = None

        self.descr_species = descr_species
        self.atomic_descriptors = None
        self.descr = None
        self.kernel = None

        if local_structures_path is None:
            self.local_structures_path = STANDARD_LOCAL_STRUCTURES_PATH
        else:
            self.local_structures_path = local_structures_path
    
    def get_unsorted_cats(self):
        """Get all names of categories, including ints for non-classified neighbour numbers.
        """
        classes = []
        for ii_id in range(self.n_classes):
            cat_string = self.id_to_cat(ii_id)
            classes.append(cat_string)
        return classes
  
    def id_to_cat(self, id_num: int):
        """Get the string name of a category for a classification integer.

        Args:
            id_num (int): integer ID to get category name for.

        Returns:
            string: Name of category for given ID int.
        """
        if id_num < len(self.low_list):
            return str(id_num)
        elif id_num < len(self.non_class_list):
            return str(id_num + (self.class_high - len(self.low_list)))
        else:
            return self.identification_list[id_num - len(self.non_class_list)]

    @staticmethod
    def _target_locator(symbols, tar_symbol="Rh"):
        """Helper function to find atom with target symbol in list of symbols.
        """
        positions = []
        for ii_symb, symb in enumerate(symbols):
            if symb == tar_symbol:
                positions.append(ii_symb)
        return positions
    
    @staticmethod
    def _rescale_atom_to_target(atoms: Atoms, rescale_target: float) -> Atoms:
        """Rescale atoms so that minimum distance in structure is equivalent to `rescale_target`.

        Returns:
            ase.Atoms: atoms with rescaled positions.
        """
        if not isinstance(rescale_target, float):
            raise TypeError("rescale_target must be of type 'float'!")

        in_atoms = atoms.copy()

        dists = in_atoms.get_all_distances()
        min_old = np.min(dists[dists>0])
        rescale_fact = rescale_target/min_old

        scaled_pos = in_atoms.get_scaled_positions()
        scaled_cell = Cell.new(in_atoms.get_cell().array*rescale_fact)
        in_atoms.set_cell(scaled_cell)

        in_atoms.set_scaled_positions(scaled_pos)

        return in_atoms

    def _soap_from_structfile(self, struct_file, ii_sf, rescale_target:float=None, **kwargs):
        """Helper function to build SOAP descriptors from file path.
        `struct_file` has to be a path which contains a file readable using `ase.io.read`.

        Args:
            struct_file (path): file to read.
            ii_sf (int): unused, only there to match general case of needing to know how many structure files are being read.
            rescale_target (float, optional): If a float is given, rescales the local structure before generating SOAPs. Defaults to None.

        Raises:
            ImportError: Raised, if there is more than one "central" atom.

        Returns:
            np.ndarray of floats: SOAP descriptors for struct_file.
        """
        struct_atoms = ase_read(struct_file)

        at_symbs = struct_atoms.get_chemical_symbols()
        at_pos = self._target_locator(at_symbs)

        if rescale_target is not None:
            struct_atoms = self._rescale_atom_to_target(struct_atoms, rescale_target)

        if not len(at_pos) == 1:
            print(struct_atoms)
            print(at_pos)
            raise ImportError("More than one Rh Atom in input file %s"%struct_file)
        else:
            at_pos = at_pos[0]

        cur_soap = self.descr.create(struct_atoms, centers=[at_pos]) # TODO: Implement a way to control how many SOAPs are used.
        return cur_soap

    def load_identifiers(self, r_cut=3.1, n_max=12, l_max=12, sigma=0.5, gamma_kernel=0.05, descr_func=None, **kwargs):
        """Loader function to create `self.identification_dict`.
        Goes through .extxyz files in directory and loads localstructures.
        For SOAP parameters see dscribe.descriptors.SOAP, for kernel see dscribe.kernels.AverageKernel.

        Args:
            r_cut (float, optional): SOAP parameter. Defaults to 3.1.
            n_max (int, optional): SOAP parameter. Defaults to 12.
            l_max (int, optional): SOAP parameter. Defaults to 12.
            sigma (float, optional): SOAP parameter. Defaults to 0.5.
            gamma_kernel (float, optional): kernel parameter. Defaults to 0.05.
            descr_func (callable, optional): Non default atomic descriptor functions. Needs to have a .create function. Defaults to None.
        """
        with open(os.path.join(self.local_structures_path, "identifiers.json"), "r") as f:
            json_dict = json.load(f)
            f.close()
        
        self.info_dict = json_dict
        self.identification_list = json_dict["identification_list"]

        for ii_struct in range(15):
            self.identification_dict[str(ii_struct)] = {"soap_descr": [], "id_num": [], "id": []}

        if descr_func is None:
            self.descr = SOAP(species=self.descr_species, r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma, periodic=False)
        else:
            self.descr = descr_func

        soap_descriptors = []
        struct_file_names = []
        struct_files = list(pathlib.Path(self.local_structures_path).glob("*.extxyz"))
        struct_files.sort()
        for ii_sf, struct_file in enumerate(struct_files):
            cur_soap = self._soap_from_structfile(struct_file, ii_sf, **kwargs)
            soap_descriptors.append(cur_soap) 

            struct_file_name = str(os.path.basename(struct_file))
            n_atoms = struct_file_name[0]

            self.identification_dict[n_atoms]["soap_descr"].append(cur_soap)
            self.identification_dict[n_atoms]["id_num"].append(ii_sf + len(self.non_class_list))
            self.identification_dict[n_atoms]["id"].append(self.identification_list[ii_sf])

        for key, value in self.identification_dict.items():
            if len(value["soap_descr"]) == 0:
                self.identification_dict[key] = None
            else:
                self.identification_dict[key]["soap_descr"] = np.asarray(self.identification_dict[key]["soap_descr"])

        self.n_classes = len(self.low_list) + len(self.identification_list) + len(self.high_list) # 0, 1, 2, 10, 11, 12, 13, 14, list, 

        self.atomic_descriptors = np.asarray(soap_descriptors)
        # self.kernel = rbf_kernel # TODO: replaces dscribe kernel, seems simpler
        self.kernel = ds_AverageKernel(metric="rbf", gamma=gamma_kernel)

    def classify(self, atoms: Atoms, mode='pre_group', n_neigh_by_class=True, ensure_position=False, **kwargs):
        """Classification function.
        Find which local structure in `self.classification_dict` is most similar to given atoms and return ID of that structure.
        Use `mode` to determine the way the classification works, available modes are:
            pre_group: only compare structures that have the same amount of nearest neighbors as `atoms`.
            class_all: compare to all local structures.

        Args:
            atoms (ase.Atoms): atoms to classify.
            mode (str, optional): Mode to use for classification. Defaults to 'pre_group'.
            n_neigh_by_class (bool, optional): Whether to get the number of neighbors from most similar class. If False, is taken from simply countin length of `atoms`.  Defaults to True.
            ensure_position (bool, optional): Make sure first atom is actually Rh. Defaults to False.

        Raises:
            ImportError: If more than on Rh atom in `atoms`.
            ValueError: If undefined mode is given.

        Returns:
            tuple of (int, int): (number of neighbors, class ID)
        """
        if mode == 'pre_group' or 'class_all':

            neighbour_len = len(atoms) - 1
            if neighbour_len <= self.class_low:
                return neighbour_len, neighbour_len
            elif neighbour_len >= self.class_high:
                return neighbour_len, neighbour_len - (self.class_high - len(self.low_list))
            else:
                if ensure_position:
                    at_symbs = atoms.get_chemical_symbols()
                    at_pos = self._target_locator(at_symbs)
                    if not len(at_pos) == 1:
                        print(atoms)
                        print(at_pos)
                        raise ImportError("More than one Rh Atom in atoms")
                
                    else:
                        at_pos = at_pos[0]
                else:
                    at_pos = 0
            soap_atoms = self.descr.create(atoms, centers=[at_pos])[np.newaxis, ...]

            if mode == 'pre_group':
                soap_base = self.identification_dict[str(neighbour_len)]["soap_descr"]
                assert soap_atoms.shape[0] == soap_base.shape[1],  "Shape mismatch in soap, %u %u"%(soap_atoms.shape[0], soap_base[0].shape[0])
                similarities = self.kernel.create(soap_base, soap_atoms)
                soap_class = np.argmax(similarities, axis=0)

                return neighbour_len, self.identification_dict[str(neighbour_len)]["id_num"][int(soap_class)]
            
            elif mode == 'class_all':
                soap_base = self.atomic_descriptors

                assert soap_atoms.shape[0] == soap_base.shape[1],  "Shape mismatch in soap, %u %u"%(soap_atoms.shape[0], soap_base[0].shape[0])
                similarities = self.kernel.create(soap_base, soap_atoms)
                soap_class = np.argmax(similarities, axis=0)
                class_id = soap_class[0] + len(self.non_class_list)

                if n_neigh_by_class:
                    neighbour_len = int(self.id_to_cat(class_id)[0])

                return neighbour_len, class_id
        else:
            raise ValueError("Mode %s is unknown. Currently available modes: \n'pre_group'\n'class_all'"%mode)

    def get_soaps(self):
        """Get all SOAP descriptors in a classifier dictionary.

        Returns:
            tuple (np.ndarray, list): SOAPs in dictionary, labels
        """
        soaps_from_classifier = []
        labels = []

        for key in self.identification_dict.keys():
            entry = self.identification_dict[key]
            if entry is not None:
                soaps_from_classifier.append(entry["soap_descr"][:, 0, :])
                labels.append(entry["id"])

        buff = soaps_from_classifier[0].copy()
        for ii_soap in range(1, len(soaps_from_classifier)):
            buff = np.append(buff, soaps_from_classifier[ii_soap], axis=0)

        soaps_from_classifier = buff.copy()
        del buff

        buff = []
        for label in labels:
            for entry in label:
                buff.append(entry)

        labels=buff
        return soaps_from_classifier, labels


class onlyCuClassifier(NeighbourClassifier):
    def __init__(self, local_structures_path=None, **kwargs):
        """Alternate classifier for classifying sites on Cu only particle.
        See sort_neigh.NeighbourClassifier for more documentation.
        """
        super().__init__(**kwargs)

        if local_structures_path is None:
            # Is already initialised in parents' init, so just get it from there
            self.local_structures_path = os.path.join(os.path.dirname(os.path.dirname(self.local_structures_path)), "localstructures_onlyCu/")
        else:
            self.local_structures_path = local_structures_path

        self.descr_species = ["Cu"] # Overwrite this so there is no more Rh

    @staticmethod
    def _target_locator(symbols, tar_symbol=None):
        # target atom should be in position 0 anyways
        return [0]

    def _soap_from_structfile(self, struct_file, ii_sf, rescale_target=None, **kwargs):
        struct_atoms = ase_read(struct_file)

        at_pos = self.info_dict["center_pos"][ii_sf] # now the information about the center needs to be contained in info dict

        if rescale_target is not None:
            struct_atoms = self._rescale_atom_to_target(struct_atoms, rescale_target)

        cur_soap = self.descr.create(struct_atoms, centers=[at_pos]) # TODO: Implement a way to control how many SOAPs are used.
        return cur_soap


class USMLClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        """Local structure classifier using unsupervised ML methods instead of local structure dictionary.
        Employs atomic descriptors in `self.descr`, a dimensionality reduction in `self.dim_red` and a clustering method into `self.clusterer`.
        """
        super().__init__(**kwargs)

        self.descr = None
        self.train_soaps = None

        self.train_particle = None
        self.dim_red = None
        self.clusterer = None

        self.descr_species = []
        
    def _train_on_data(self, data, dim_red=None, clusterer=None):
        """Train the dimensionality reduction on data.

        Args:
            data (np.ndarray): Training data for dimensionality reduction shaped (n_data x n_features).
            dim_red (callable, optional): Function for dimensionality reduction. If None is given uses sklean.decomposition.PCA. Needs to have a .fit_transform method. Defaults to None.
            clusterer (callable, optional): Clusterer. If None is given defaults to sklearn.cluster.KMeans. Needs to have a .fit_predict method. Defaults to None.

        Returns:
            np.ndarray of ints: cluster number for each entry in training data, shaped (n_data,).
        """
        if dim_red is None:
            self.dim_red = PCA()
        else:
            self.dim_red = dim_red

        if clusterer is None:
            self.clusterer = KMeans()
        else:
            self.clusterer = clusterer
    
        reduced = self.dim_red.fit_transform(data)
        n_clust = self.clusterer.fit_predict(reduced)

        return n_clust

    def train_on_particle(self, particle: Atoms, dim_red=None, clusterer=None, descr_species=["Rh", "Cu"], r_cut=3.1, n_max=12, l_max=12, sigma=0.5, descr_func=None, **kwargs):
        """Train the classification on an ase.Atoms object using either SOAP or a function found in `dim_red`.

        Args:
            particle (ase.Atoms): Training particle.
            dim_red (callable, optional): Function for dimensionality reduction. If None is given uses sklean.decomposition.PCA, needs to have a .fit_transform method. Defaults to None.
            clusterer (callable, optional): Clusterer. If None is given defaults to sklearn.cluster.KMeans. Needs to have a .fit_predict method. Defaults to None.
            descr_species (list, optional): SOAP parameter. Defaults to ["Rh", "Cu"].
            r_cut (float, optional): SOAP parameter. Defaults to 3.1.
            n_max (int, optional): SOAP parameter. Defaults to 12.
            l_max (int, optional): SOAP parameter. Defaults to 12.
            sigma (float, optional): SOAP parameter. Defaults to 0.5.
            descr_func (callable, optional): Atomic descriptor function to overwrite SOAP. If none is given, defaults to SOAP. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.descr_species = descr_species
        self.train_particle = particle.copy()

        if descr_func is None:
            self.descr = SOAP(species=self.descr_species, r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma, periodic=False)
        else:
            self.descr = descr_func
        soaps = self.descr.create(particle)

        return self._train_on_data(soaps, dim_red=dim_red, clusterer=clusterer)
    
    def save(self, file_name="USMLClassifier.pickle", overwrite=False):
        """Saves ML Classifier to file via pickle.

        Args:
            file_name (str, optional): File name to save into. Defaults to "USMLClassifier.pickle".
            overwrite (bool, optional): Whether or not to overwrite file if already exists. Defaults to False.

        Raises:
            FileExistsError: File exists already and overwrite flag is False.
        """
        if not overwrite and os.path.isfile(file_name):
            raise FileExistsError("File %s already exists. To overwrite its contents set overwrite-flag to 'True'.")
            
        with open(file_name, "wb") as f:
            pickle.dump(self.train_particle, f)
            pickle.dump(self.descr, f)
            pickle.dump(self.dim_red, f)
            pickle.dump(self.clusterer, f)
            f.close()
    
    @classmethod
    def load_from_file(cls, file_name, **kwargs):
        """Class constructor from save file created via `sort_neigh.USMLClassifier.save`.

        Args:
            file_name (str or path): File Location to load from.
        """
        out_classifier = cls(**kwargs)

        with open(file_name, "rb") as f:
            out_classifier.train_particle = pickle.load(f)
            out_classifier.descr = pickle.load(f)
            out_classifier.dim_red = pickle.load(f)
            out_classifier.clusterer = pickle.load(f)
            f.close()
        
        return out_classifier


    def classify(self, atoms: Atoms, ensure_position=False, index=None):
        """Classification method to find cluster number of given atomic environment.
        Builds atomic descriptor, applies dimensionality reduction and performs clustering to return cluster number.

        Args:
            atoms (Atoms): atoms to be classified.
            ensure_position (bool, optional): Unused. Defaults to False.
            index (int, optional): Index of "central" atom. Defaults to None.

        Returns:
            int: Number of cluster / Classification.
        """
        if index is None:
            soaps = self.descr.create(atoms)
        else:
            soaps = self.descr.create(atoms)[index, :]
        if len(soaps.shape) < 2:
            soaps = soaps[np.newaxis, :]
        reduced = self.dim_red.transform(soaps)
        n_clust = self.clusterer.predict(reduced)

        return n_clust

class NeighbourSort():
    def __init__(self, classifier=None, **kwargs):
        """Class to handle loading of trajectory and classifying each relevant atom in it.

        Args:
            classifier (sort_neigh.BaseClassifier, optional): Classifier to classify each site with. Defaults to None.
        """
        if classifier is None:
            self.classifier = NeighbourClassifier(**kwargs)
        else:
            self.classifier = classifier
        self.classifier.load_identifiers(**kwargs)

        self.particle_trajectory = None

        self.cat_counter = None
        self.or_file = None
        self.work_dir_path = None
        self.timesteps = None

    def load_particle(self, file_path, make_folders=False, timesteps=None, **kwargs):
        """Load trajectory from file path.
        `file_path` needs to be readable via ase.io.read.

        Args:
            file_path (str or path): Trajectory file, nees to be readable via ase.io.read.
            make_folders (bool, optional): Whether or not a folder structure should be created to save each timestep in. Defaults to False.
            timesteps (int, optional): Max number of timesteps to analyse. If None are given, does every available timestep in trajectory. Defaults to None.

        Raises:
            ValueError: If there are more timesteps in timesteps than in trajectory.
        """
        if make_folders:
            self.init_folder_structure(file_path=file_path, timesteps=timesteps, **kwargs)

        else:
            self.or_file_path = os.path.abspath(file_path)
            self.particle_trajectory = ase_read(file_path, index=':') # This loads the whole trajectory into memory, possibly a problem but should be fine

            self.timesteps = len(self.particle_trajectory)
            if timesteps is not None:
                if timesteps <= self.timesteps:
                    self.timesteps = timesteps
                else:
                    raise ValueError("There is only %u timesteps in particle, therefore cannot analyze %u steps."%(self.timesteps, timesteps))

            for n_particle in range(self.timesteps):
                self.particle_trajectory[n_particle] = ase_sort(self.particle_trajectory[n_particle]) # So Rh is always in the last indices

    def init_folder_structure(self, file_path, n_head=9, n_tail=0, n_atoms_in_part=1400, timesteps=None, out_dir=None):
        """Semi outdated function for handreading trajectory and creating folder structure.
        Folder structure will be constructed as eg.:

        neigh_sort_out/ # or out_dir if given
            ts_0/
                9/
                10/
                12/
            ts_1/
            ...
            ts_ntimesteps/

        where the ts_x/ directories contain the trajectory snapshot as well as subdirectories containing snapshots of the local environments depending on the number of neighbours.

        Args:
            file_path (str or path): target file
            n_head (int, optional): Number of lines before each timestep. Defaults to 9.
            n_tail (int, optional): Number of lines after each timestep. Defaults to 0.
            n_atoms_in_part (int, optional): Number of atoms per timestep. Defaults to 1400.
            timesteps (int, optional): Number of timesteps in trajectory. Defaults to None.
            out_dir (str or path, optional): name of output trajectory. Defaults to None.
        """
        if timesteps is None:
            n_lines = sum(1 for line in open(file_path, 'r'))
            timesteps = n_lines/(n_atoms_in_part+n_head+n_tail)
            if not float(timesteps).is_integer():
                raise ValueError(
                    "Number of lines in %s not fully divisible by n_atoms_in_part + n_head + n_tail (%u + %u + %u)"%(
                        file_path, n_atoms_in_part, n_head, n_tail
                    )
                )
            timesteps = int(timesteps)
            self.timesteps = timesteps
        else:
            if not isinstance(timesteps, int):
                raise ValueError("timesteps must be of type 'int'.")
            self.timesteps = int(timesteps)
        
        self.or_file_path = os.path.abspath(file_path)
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(self.or_file_path), "neigh_sort_out/")
        
        self.work_dir_path = os.path.abspath(out_dir)

        if not os.path.isdir(self.work_dir_path):
            os.mkdir(self.work_dir_path)
        else:
            return # Possibly change this, but could just be set so it finds folders
            raise FileExistsError("The output directory:\n%s\nalready exists, please move or rename it for NeighbourSort to work."%self.work_dir_path)

        with open(file_path, 'r') as f:
            f_lines = f.read().split('\n')
            f.close()

        lines_per_part = n_head + n_tail + n_atoms_in_part

        for step in range(timesteps):
            line_start = lines_per_part * step
            line_end = lines_per_part * (step + 1)
            
            new_lines = f_lines[line_start:line_end]
            particles = new_lines[n_head:n_head+n_atoms_in_part]
            particles.sort() # TODO: Somehow keep track of particles?
            new_lines[n_head:n_head+n_atoms_in_part] = particles
            
            cur_dir = os.path.join(self.work_dir_path, "ts_%u/"%step)
            os.mkdir(cur_dir)
            cur_file = os.path.join(cur_dir, "cunano_%u.lammpstrj"%step)
            with open(cur_file, 'w') as f:
                f.write('\n'.join(new_lines))
                f.close()

    def sort_cat_counter(self, cat_counter=None):
        """Sort a category counter according to the number of neighbours each category has.
        To do this, the categories in classifier need to be named eg. 3_..., 4_... with the number of neighbours in front.

        Args:
            cat_counter (np.ndarray, optional): Category counter to sort, cat_counter.shape[-1] needs to be len(self.categories). If None is given, uses self.cat_counter. Defaults to None.

        Returns:
            tuple (list, np.ndarray): (sorted categories, sorted category counter)
        """
        if cat_counter is None:
            cat_counter = self.cat_counter

        classes = []
        particle_num = []
        for ii_id in range(cat_counter.shape[-1]):
            cat_string = self.classifier.id_to_cat(ii_id)
            classes.append(cat_string)

            num = int(cat_string.split('_', 1)[0])
            particle_num.append(num)

        sort_arr = np.argsort(np.asarray(particle_num))
        sort_arr = np.asarray(sort_arr, np.int32)

        sorted_classes = []
        for sort_ind in sort_arr:
            sorted_classes.append(classes[sort_ind])
            
        sorted_part_count = np.take(cat_counter, sort_arr, axis=1)
        return sorted_classes, sorted_part_count

    def create_local_structure(self, cutoff_mult=0.9, last_n=14, **kwargs):
        """Generate classification for each timestep in trajectory.

        Args:
            cutoff_mult (float, optional): Cutoff multiplier for neighbor list. Defaults to 0.9.
            last_n (int, optional): Number of Special atoms to find, will be placed at botton of each timestep. Defaults to 14.

        Raises:
            ValueError: If there is no target particle found.

        Returns:
            np.ndarray of ints: Resulting categorisation shaped (n_timesteps, n_classes)
        """
        if self.particle_trajectory is not None:
            self.cat_counter = np.zeros(shape=(self.timesteps, self.classifier.n_classes), dtype=np.int32)

            init_particle = self.particle_trajectory[0]
            particle_len = len(init_particle)
            particle_range = (particle_len - np.arange(last_n)[::-1])-1

            cut_off = natural_cutoffs(init_particle, mult=cutoff_mult)
            neighbour_list = NeighborList(cut_off, bothways=True, self_interaction=False)
            neighbour_list.update(init_particle)

            for step in self.progressbar(range(self.timesteps), "At Timestep:", size=40):
                cur_particle = self.particle_trajectory[step]
                neighbour_list.update(cur_particle)

                for index in particle_range:
                    neighbour_indices, dists = neighbour_list.get_neighbors(index)
                    neighbour_indices = np.append(np.array([index]), neighbour_indices, axis=0)

                    neighbour_particle = cur_particle[neighbour_indices]
                    neighbour_particle.symbols[:] = 'Cu'
                    neighbour_particle.symbols[0] = 'Rh'
                    n_neighbours = len(neighbour_particle) - 1
                    
                    n_neigh, class_id = self.classifier.classify(neighbour_particle, **kwargs)
                    self.cat_counter[step, class_id] += 1

            return self.cat_counter
        else:
            if self.work_dir_path is None:
                raise ValueError("""No particle loaded in self.particle_trajectory or self.work_dir_path.
                To create local structure use load_particle to initialise the target particle either in memory or subfolders.
                """)
            return self._locstruct_from_folder(last_n=last_n, cutoff_mult=cutoff_mult, **kwargs)


    def _locstruct_from_folder(self, last_n=14, create_subfolders=True, cutoff_mult=0.9, **kwargs):
        """Does the same as create_local_structure(), but needs to be used if folders had been created using init_folder_structure.
        """
        self.cat_counter = np.zeros(shape=(self.timesteps, self.classifier.n_classes), dtype=np.int32)
        for step in self.progressbar(range(self.timesteps), "At Timestep:", size=40):

            cur_dir = pathlib.Path(os.path.join(self.work_dir_path, "ts_%u/"%step))
            files = list(cur_dir.glob("*.lammpstrj"))
            for cur_file in files:
                full_particle = ase_read(cur_file)

                particle_len = len(full_particle)
                particle_range = (particle_len - np.arange(last_n)[::-1])-1

                cut_off = natural_cutoffs(full_particle, mult=cutoff_mult)
                neighbour_list = NeighborList(cut_off, bothways=True, self_interaction=False)
                neighbour_list.update(full_particle)

                class_atoms = []
                for index in particle_range:
                    neighbour_indices, trash = neighbour_list.get_neighbors(index)
                    neighbour_indices = np.append(np.array([index]), neighbour_indices, axis=0)
                    neighbour_particle = full_particle[neighbour_indices]
                    class_atoms.append(neighbour_particle)

                    n_neighbours = len(neighbour_particle) - 1
                    
                    n_neigh, class_id = self.classifier.classify(neighbour_particle, **kwargs)
                    class_cat = self.classifier.id_to_cat(class_id)
                    self.cat_counter[step, class_id] += 1
                    
                    if create_subfolders:
                        count_dir = os.path.join(cur_dir, "%u/"%n_neighbours)
                        if not os.path.isdir(count_dir):
                            os.mkdir(count_dir)

                        if len(class_cat) > 3:
                            cat_dir = os.path.join(count_dir, class_cat)
                            tar_path = os.path.join(cat_dir, "%u_neighbors"%n_neighbours)
                            if not os.path.isdir(cat_dir):
                                os.mkdir(cat_dir)
                        else:
                            tar_path = os.path.join(count_dir, "%u_neighbors"%n_neighbours)
                        tar_path = str(tar_path) + "_%u.extxyz"%(self.cat_counter[step, class_id])
                        neighbour_particle.write(tar_path, format='extxyz')
        
        return self.cat_counter

    @staticmethod
    def remove_n_surf(file_name, n_remove=64, out_file_name=None):
        """Remove n random atoms from surface of nanoparticle.
        Target file needs to be a single snapshot of trajectory, otherwise only first snapshot will be used.

        Args:
            file_name (str or path): Target file.
            n_remove (int, optional): Number of surface atoms to remove. Defaults to 64.
            out_file_name (str or path, optional): File name to save to. Defaults to None.

        Raises:
            ValueError: If target file has less atoms than n_remove.

        Returns:
            str: name of outgoing save file.
        """
        file_name = os.path.abspath(file_name)

        if out_file_name is None:
            out_file_name = file_name.split(".")
            out_file_name[-2] = out_file_name[-2] + "_rem%u"%n_remove
            out_file_name = ".".join(out_file_name)

        out_format = file_name.split(".")[-1]
        if out_format == "lammps-data":
            original_particle = ase_read(file_name, format=out_format, style='atomic')
        else:
            original_particle = ase_read(file_name, format=out_format)

        n_atoms = len(original_particle)

        # set up neighbourlist
        cut_off = natural_cutoffs(original_particle, mult=0.9)
        neighbour_list = NeighborList(cut_off, bothways=True, self_interaction=False)
        neighbour_list.update(original_particle)

        # Determine number of neighbours for each atom in nanoparticle
        n_neighs = [] 
        for nn_atom in range(n_atoms):
            neighbour_indices, trash = neighbour_list.get_neighbors(nn_atom)
            neighbour_indices = np.append(np.array([nn_atom]), neighbour_indices, axis=0)
            n_neighs.append(len(neighbour_indices))
        n_neighs = np.asarray(n_neighs, dtype=np.int32)

        is_surf = n_neighs <= 9
        is_surf_inds = np.arange(n_atoms)[is_surf]
        
        n_surf = np.sum(np.asarray(is_surf, dtype=np.int32))
        if n_surf < n_remove:
            raise ValueError(
                "Cannot remove %u surface atoms from particle with only %u atoms on surface"%(n_remove, n_surf)
            )

        # Figure out which indices to remove
        rng = default_rng()
        remove_inds = rng.choice(is_surf_inds, size=n_remove, replace=False)
        remove_inds = np.sort(remove_inds)[::-1]

        # Remove from back, but seems to not be necessary
        out_particle = original_particle.copy()
        del out_particle[remove_inds]

        out_particle.write(out_file_name, format=out_format)
        return out_file_name
        
    def sort_save_cat(self, file_name=None, cat_counter=None):
        """Sort category count and save to .txt files.

        Args:
            file_name (str or path, optional): File name for saving categories into. Defaults to None.
            cat_counter (np.ndarray, optional): Target category counter. If None is given, self.cat_counter is saved. Defaults to None.

        Returns:
            str: Name of saved file.
        """
        if file_name is None:
            file_name = os.path.join(os.path.dirname(self.or_file_path), "part_count.txt")

        if cat_counter is None:
            cat_counter = self.cat_counter

        sorted_classes, sorted_cat_counter = self.sort_cat_counter(cat_counter=cat_counter)
        header = "Saved category counter for %s.\ntimestep "%self.work_dir_path
        for sorted_class in sorted_classes:
            header += "%s "%sorted_class
        
        print_cat = np.zeros((self.timesteps, self.classifier.n_classes+1), dtype=np.int32)
        print_cat[:, 0] = np.arange(self.timesteps)
        print_cat[:, 1:] = sorted_cat_counter

        np.savetxt(file_name, print_cat, header=header, fmt='%u')
        return file_name

    @staticmethod
    def load_sort_cat(file_name):
        """Load category counter from saved file via sort_neigh.NeighbourSort.sort_save_cat.

        Args:
            file_name (str or path): Target file name.

        Returns:
            tuple (np.ndarray, np.ndarray, list): (category counter, timesteps, categories)
        """
        with open(file_name, "r") as f:
            cat_line = f.read().split('\n')[1]
            cat_line = cat_line.split(' ')
            sorted_cats = cat_line[2:-1] # Cut off hash symbol, timesteps label and final whitespace
            f.close()

        cat_counter = np.loadtxt(file_name, dtype=np.int32)

        return cat_counter[:, 1:], cat_counter[:, 0], sorted_cats
    
    @staticmethod
    def block_average(counts, block_size=10, mode='blocks'):
        """Calculate variance of sites using blocked averages.

        Args:
            counts (np.ndarray): array of dim 2, shaped timesteps x categories
            block_size (int, optional): Number of elements per block. Defaults to 10.
            mode (str, optional): Mode to calculate blocks.
            Currently available:
                -> 'blocks': Take block_size consecutive timesteps from counts.
            Defaults to 'blocks'.

        Returns:
            np.ndarray: floats shaped categories, with block error in each element.
        """
        # TODO: add strided mode
        if not (len(counts.shape)) == 2:
            raise ValueError("Dimension of variable counts needs to be exactly 2: timesteps x categories")

        if mode == 'blocks':
            n_timesteps = counts.shape[0]
            n_blocks = n_timesteps // block_size

            total_sites = np.sum(counts, axis=0)

            blocked_cats = counts[:n_blocks*block_size, :].reshape((n_blocks, block_size, counts.shape[-1]))
            block_totals = np.sum(blocked_cats, axis=-2)

            brackets = (block_totals - (total_sites/float(n_blocks)))**2
            exp_val = np.sum(brackets, axis=-2) / float(n_blocks)

            err = n_blocks * np.sqrt(exp_val/float(n_blocks))
            return err
        else:
            raise ValueError("Mode '%s' is not a valid mode for forming the blocks."%mode)

    @staticmethod
    def progressbar(it, prefix="", size=60, file=sys.stdout):
        count = len(it)
        def show(j):
            x = int(size*j/count)
            file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
            file.flush()        
        show(0)
        for i, item in enumerate(it):
            yield item
            show(i+1)
        file.write("\n")
        file.flush()

    @staticmethod
    def plot_dist(sorted_classes, sorted_cat_counter, show_plot=True):
        """Fast plotting function to get barchart of sorted categories and class list.
        """
        x_plot = np.arange(sorted_cat_counter.shape[-1])
    
        sort_total_count = np.sum(sorted_cat_counter, axis=0)
        
        sort_not_zero = sort_total_count != 0

        fig, axis = plt.subplots(figsize=(12, 8))
        axis.set_title('Categorized Surface Atoms over %u Timesteps'%(sorted_cat_counter.shape[0]))
        axis.bar(x_plot, height=sort_total_count)
        axis.set_xticks(x_plot, sorted_classes, rotation=320, ha='left')
        axis.set_yscale('log')
        axis.grid(axis='y')

        fig2, axis2 = plt.subplots(figsize=(20, 7))
        im = axis2.imshow(sorted_cat_counter.T[:, :], aspect='auto', cmap="viridis")
        axis2.set_xlabel('timestamp')
        axis2.set_ylabel('category')
        axis2.set_yticks(x_plot, sorted_classes)
        ax_cb = fig2.colorbar(im)
        
        not_zero_classes = []
        for ii_nz, not_zero in enumerate(sort_not_zero):
            if not_zero:
                not_zero_classes.append(sorted_classes[ii_nz])
        x_plot_not_zero = np.arange(len(not_zero_classes))
        # print(len(not_zero_classes), x_plot_not_zero.shape)

        fig3, axis3 = plt.subplots(figsize=(20, 7))
        im = axis3.imshow(sorted_cat_counter.T[sort_not_zero, :], aspect='auto', cmap="viridis")
        axis3.set_xlabel('timestamp')
        axis3.set_ylabel('category')
        axis3.set_yticks(x_plot_not_zero, not_zero_classes)
        ax_cb = fig3.colorbar(im)
        
        if show_plot:
            plt.show()
        else:
            return [fig, fig2, fig3], [axis, axis2, axis3]
