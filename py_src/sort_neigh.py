import os
import pathlib
import numpy as np
import json
import matplotlib.pyplot as plt

from ase.io import write as ase_write
from ase.io import read as ase_read
from ase.neighborlist import natural_cutoffs, NeighborList

from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel as ds_AverageKernel
from sklearn.metrics.pairwise import rbf_kernel

STANDARD_LOCAL_STRUCTURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),"src/localstructures/")

class NeighbourClassifier():
    def __init__(self, local_structures_path=None, first_class=3, last_class=9, non_class_max=14):
        self.class_low = first_class - 1
        self.class_high = last_class + 1
        self.low_list = [ii for ii in range(0, self.class_low + 1)]
        self.high_list = [ii for ii in range(self.class_high, non_class_max+1)]
        self.non_class_list = self.low_list + self.high_list

        self.local_structures_path = None
        self.identification_list = []
        self.identification_dict = {}
        self.n_classes = None

        self.soap = None
        self.kernel = None
        self.soap_descriptors = []

        if local_structures_path is None:
            os.path.abspath(__file__)
            self.local_structures_path = STANDARD_LOCAL_STRUCTURES_PATH
        else:
            self.local_structures_path = local_structures_path
    
    def load_identifiers(self, rcut=3.1, nmax=12, lmax=12, sigma=0.5, gamma_kernel=0.05):
        with open(os.path.join(self.local_structures_path, "identifiers.json"), "r") as f:
            json_dict = json.load(f)
            f.close()
        
        self.identification_list = json_dict["identification_list"]

        for ii_struct in range(15):
            self.identification_dict[str(ii_struct)] = {"soap_descr": [], "id_num": [], "id": []}

        self.soap = SOAP(species=["Rh", "Cu"], rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma, periodic=False)
        self.rh_cu_loc = self.soap.get_location(("Cu", "Rh"))

        soap_descriptors = []
        struct_file_names = []
        struct_files = list(pathlib.Path(self.local_structures_path).glob("*.extxyz"))
        struct_files.sort()
        for ii_sf, struct_file in enumerate(struct_files):
            struct_atoms = ase_read(struct_file)

            at_symbs = struct_atoms.get_chemical_symbols()
            at_pos = self._get_positions(at_symbs)
            if not len(at_pos) == 1:
                raise ImportError("More than one Rh Atom in input file %s"%struct_file)
            else:
                at_pos = at_pos[0]
            cur_soap = self.soap.create(struct_atoms)[at_pos, :][np.newaxis, ...]
            soap_descriptors.append(cur_soap) # TODO: remove

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

        self.soap_descriptors = np.asarray(soap_descriptors) # TODO: remove
        # self.kernel = rbf_kernel # TODO: replaces dscribe kernel, seems simpler
        self.kernel = ds_AverageKernel(metric="rbf", gamma=gamma_kernel)
        # self.kernel = ak.create(self.soap_descriptors) # TODO: see how similar they are to one another

    def classify(self, atom, mode='pre_group', n_neigh_by_class=True):
        if mode == 'pre_group' or 'class_all':

            neighbour_len = len(atom) - 1
            if neighbour_len <= self.class_low:
                return neighbour_len, neighbour_len
            elif neighbour_len >= self.class_high:
                return neighbour_len, neighbour_len - (self.class_high - len(self.low_list))
            else:
                at_symbs = atom.get_chemical_symbols()
                at_pos = self._get_positions(at_symbs)
                if not len(at_pos) == 1:
                    raise ImportError("More than one Rh Atom in atom")
                else:
                    at_pos = at_pos[0]
                
            soap_atom = self.soap.create(atom)[at_pos, :][np.newaxis, np.newaxis, ...]

            if mode == 'pre_group':
                soap_base = self.identification_dict[str(neighbour_len)]["soap_descr"]
                assert soap_atom.shape[0] == soap_base.shape[1],  "Shape mismatch in soap, %u %u"%(soap_atom.shape[0], soap_base[0].shape[0])
                
                similarities = self.kernel.create(soap_base, soap_atom)
                soap_class = np.argmax(similarities, axis=0)

                return neighbour_len, self.identification_dict[str(neighbour_len)]["id_num"][int(soap_class)]
            
            elif mode == 'class_all':
                soap_base = self.soap_descriptors

                assert soap_atom.shape[0] == soap_base.shape[1],  "Shape mismatch in soap, %u %u"%(soap_atom.shape[0], soap_base[0].shape[0])
                similarities = self.kernel.create(soap_base, soap_atom)
                
                soap_class = np.argmax(similarities, axis=0)
                if n_neigh_by_class:
                    neighbour_len = self.id_to_cat(soap_class[0] + len(self.non_class_list))

                return neighbour_len, soap_class[0] + len(self.non_class_list)
    
    @staticmethod
    def _get_positions(symbols, tar_symbol="Rh"):
        positions = []
        for ii_symb, symb in enumerate(symbols):
            if symb == tar_symbol:
                positions.append(ii_symb)
        return positions

    def id_to_cat(self, id_num):
        if id_num < len(self.low_list):
            return str(id_num)
        elif id_num < len(self.non_class_list):
            return str(id_num + (self.class_high - len(self.low_list)))
        else:
            return self.identification_list[id_num - len(self.non_class_list)]


class NeighbourSort():
    def __init__(
        self, local_structures_path=None, first_class=3, last_class=9, non_class_max=14,
        rcut=3.1, nmax=12, lmax=12, sigma=0.5, gamma_kernel=0.05
    ):
        if local_structures_path is None:
            os.path.abspath(__file__)
            self.local_structures_path = STANDARD_LOCAL_STRUCTURES_PATH
        else:
            self.local_structures_path = local_structures_path
        
        self.classifier = NeighbourClassifier(self.local_structures_path, first_class, last_class, non_class_max)
        self.classifier.load_identifiers(rcut, nmax, lmax, sigma, gamma_kernel)

        self.cat_counter = None
        self.or_file = None
        self.work_dir_path = None
        self.timesteps = None

    def init_folder_structure(self, file_path, n_head=9, n_tail=0, n_atoms_in_part=1400, timesteps=100, out_dir=None):
        self.timesteps = timesteps

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
            particles.sort()
            new_lines[n_head:n_head+n_atoms_in_part] = particles
            
            cur_dir = os.path.join(self.work_dir_path, "ts_%u/"%step)
            os.mkdir(cur_dir)
            cur_file = os.path.join(cur_dir, "cunano_%u.lammpstrj"%step)
            with open(cur_file, 'w') as f:
                f.write('\n'.join(new_lines))
                f.close()

    def sort_cat_counter(self, cat_counter=None):
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

    def create_local_structure(self, last_n=14, create_subfolders=True, mode="pre_group"):
        self.cat_counter = np.zeros(shape=(self.timesteps, self.classifier.n_classes), dtype=np.int32)

        for step in range(self.timesteps):

            cur_dir = pathlib.Path(os.path.join(self.work_dir_path, "ts_%u/"%step))
            files = list(cur_dir.glob("*.lammpstrj"))
            for cur_file in files:
                full_particle = ase_read(cur_file)

                particle_len = len(full_particle)
                particle_range = (particle_len - np.arange(last_n)[::-1])-1

                cut_off = natural_cutoffs(full_particle, mult=0.9)
                neighbour_list = NeighborList(cut_off, bothways=True, self_interaction=True)
                neighbour_list.update(full_particle)

                class_atoms = []
                for index in particle_range:
                    neighbour_indices, trash = neighbour_list.get_neighbors(index)
                    neighbour_particle = full_particle[neighbour_indices[:-1]]
                    class_atoms.append(neighbour_particle)

                    n_neighbours = len(neighbour_particle) - 1
                    
                    n_neigh, class_id = self.classifier.classify(neighbour_particle, mode=mode)
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
        
    def sort_save_cat(self, file_name=None, cat_counter=None):
        if file_name is None:
            file_name = os.path.join(os.path.dirname(self.or_file_path), "part_count.txt")

        if cat_counter is None:
            cat_counter = self.cat_counter

        sorted_classes, sorted_cat_counter = self.sort_cat_counter(cat_counter=cat_counter)
        header = """Saved category counter for %s.
        timestep 
        """%self.work_dir_path
        for sorted_class in sorted_classes:
            header += """%s """%sorted_class
        
        print_cat = np.zeros((self.timesteps, self.classifier.n_classes+1), dtype=np.int32)
        print_cat[:, 0] = np.arange(self.timesteps) + 1
        print_cat[:, 1:] = sorted_cat_counter

        np.savetxt(file_name, print_cat, header=header)
        return file_name

    def load_sort_cat(self, file_name):
        with open(file_name, "r") as f:
            sorted_cats = f.read().split('\n')[1].split(' ')[3:]
            f.close()

        cat_counter = np.loadtxt(file_name, dtype=np.int32)

        return cat_counter[:, 1:], cat_counter[:, 0], sorted_cats

    @staticmethod
    def plot_dist(sorted_classes, sorted_cat_counter):
        x_plot = np.arange(sorted_cat_counter.shape[-1])
    
        sort_total_count = np.sum(sorted_cat_counter, axis=0)
        
        sort_not_zero = sort_total_count != 0

        fig, axis = plt.subplots(figsize=(12, 8))
        axis.set_title('Categorized Surface Atoms over 100 Timesteps')
        axis.bar(x_plot, height=sort_total_count)
        axis.set_xticks(x_plot, sorted_classes, rotation=320)
        axis.set_yscale('log')
        axis.grid(axis='y')

        fig2, axis2 = plt.subplots(figsize=(150, 7))
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

        fig3, axis3 = plt.subplots(figsize=(150, 7))
        im = axis3.imshow(sorted_cat_counter.T[sort_not_zero, :], aspect='auto', cmap="viridis")
        axis3.set_xlabel('timestamp')
        axis3.set_ylabel('category')
        axis3.set_yticks(x_plot_not_zero, not_zero_classes)
        ax_cb = fig3.colorbar(im)
        
        plt.show()
