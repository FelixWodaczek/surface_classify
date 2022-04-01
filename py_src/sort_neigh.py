import os
import pathlib
from re import S
import numpy as np
import json

from ase.io import write as ase_write
from ase.io import read as ase_read
from ase.neighborlist import natural_cutoffs, NeighborList

from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel as ds_AverageKernel
from sklearn.metrics.pairwise import rbf_kernel
from pyparsing import autoname_elements

STANDARD_LOCAL_STRUCTURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),"src/localstructures/")

class NeighbourClassifier():
    def __init__(self, local_structures_path=None):
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
        
        with open(os.path.join(self.local_structures_path, "identifiers.json"), "r") as f:
            json_dict = json.load(f)
            f.close()
        
        self.identification_list = json_dict["identification_list"]

        for ii_struct in range(15):
            self.identification_dict[str(ii_struct)] = {"soap_descr": [], "id_num": [], "id": []}

        self.soap = SOAP(species=["Rh", "Cu"], rcut=3.5, nmax=12, lmax=12, sigma=0.1, periodic=False)
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
            self.identification_dict[n_atoms]["id_num"].append(ii_sf + 8)
            self.identification_dict[n_atoms]["id"].append(self.identification_list[ii_sf])

        for key, value in self.identification_dict.items():
            if len(value["soap_descr"]) == 0:
                self.identification_dict[key] = None
            else:
                self.identification_dict[key]["soap_descr"] = np.asarray(self.identification_dict[key]["soap_descr"])

        self.n_classes = 3 + len(self.identification_list) + 5 # 0, 1, 2, 10, 11, 12, 13, 14, list, 

        self.soap_descriptors = np.asarray(soap_descriptors) # TODO: remove
        # self.kernel = rbf_kernel # TODO: replaces dscribe kernel, seems simpler
        self.kernel = ds_AverageKernel(metric="rbf", gamma=0.05)
        # self.kernel = ak.create(self.soap_descriptors) # TODO: see how similar they are to one another

    def classify(self, atom):
        neighbour_len = len(atom) - 1
        if neighbour_len < 3:
            return neighbour_len, neighbour_len
        elif neighbour_len > 9:
            return neighbour_len, neighbour_len - 7
        else:
            at_symbs = atom.get_chemical_symbols()
            at_pos = self._get_positions(at_symbs)
            if not len(at_pos) == 1:
                raise ImportError("More than one Rh Atom in atom")
            else:
                at_pos = at_pos[0]

            soap_atom = self.soap.create(atom)[at_pos, :][np.newaxis, np.newaxis, ...]
            soap_base = self.identification_dict[str(neighbour_len)]["soap_descr"]
            soap_base = self.soap_descriptors
            assert soap_atom.shape[0] == soap_base.shape[1],  "Shape mismatch in soap, %u %u"%(soap_atom.shape[0], soap_base[0].shape[0])
            similarities = self.kernel.create(soap_base, soap_atom)
            
            soap_class = np.argmax(similarities, axis=0)
            # print(neighbour_len, similarities)
            # print(self.identification_dict[str(neighbour_len)]["id"][int(soap_class)])
            return neighbour_len, soap_class[0] + 8
            return neighbour_len, self.identification_dict[str(neighbour_len)]["id_num"][int(soap_class)]
    
    @staticmethod
    def _get_positions(symbols, tar_symbol="Rh"):
        positions = []
        for ii_symb, symb in enumerate(symbols):
            if symb == tar_symbol:
                positions.append(ii_symb)
        return positions

    def id_to_cat(self, id_num):
        if id_num < 3:
            return str(id_num)
        elif id_num < 8:
            return str(id_num + 7)
        else:
            return self.identification_list[id_num - 8]


class NeighbourSort():
    def __init__(self, local_structures_path=None):
        if local_structures_path is None:
            os.path.abspath(__file__)
            self.local_structures_path = STANDARD_LOCAL_STRUCTURES_PATH
        else:
            self.local_structures_path = local_structures_path
        
        self.classifier = NeighbourClassifier(self.local_structures_path)
        self.cat_counter = None
        self.or_file = None
        self.work_dir_path = None
        self.timesteps = None

    def init_folder_structure(self, file_path, n_head=9, n_tail=0, n_particles=1400, timesteps=100, out_dir=None):
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

        lines_per_part = n_head + n_tail + n_particles

        for step in range(timesteps):
            line_start = lines_per_part * step
            line_end = lines_per_part * (step + 1)

            
            new_lines = f_lines[line_start:line_end]
            particles = new_lines[n_head:n_head+n_particles]
            particles.sort()
            new_lines[n_head:n_head+n_particles] = particles
            
            cur_dir = os.path.join(self.work_dir_path, "ts_%u/"%step)
            os.mkdir(cur_dir)
            cur_file = os.path.join(cur_dir, "cunano_%u.lammpstrj"%step)
            with open(cur_file, 'w') as f:
                f.write('\n'.join(new_lines))
                f.close()


    def create_local_structure(self, last_n=14, create_subfolders=True):
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
                    
                    n_neigh, class_id = self.classifier.classify(neighbour_particle)
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
        # TODO: subfolder structure
                    