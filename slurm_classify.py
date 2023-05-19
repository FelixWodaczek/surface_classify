import os
import re
from argparse import ArgumentParser
from dscribe.descriptors import LMBTR, SOAP
from ase.io import read as ase_read
from ase.build import sort as ase_sort
from ase.neighborlist import natural_cutoffs, NeighborList
import numpy as np
import py_src.sort_neigh as sort_neigh
from sklearn.decomposition import PCA

def _descriptors_from_classifier(classifier):
    soaps_from_classifier = []

    for key in classifier.identification_dict.keys():
        entry = classifier.identification_dict[key]
        if entry is not None:
            soaps_from_classifier.append(entry["soap_descr"][:, 0, :])

    buff = soaps_from_classifier[0].copy()
    for ii_soap in range(1, len(soaps_from_classifier)):
        buff = np.append(buff, soaps_from_classifier[ii_soap], axis=0)

    soaps_from_classifier = buff.copy()
    return soaps_from_classifier

class _PCA_quick:
    def __init__(self, descr, pca: PCA):
        # Small class to be able to use "create"
        self.descr = descr
        self.pca = pca

    def create(self, atom, positions):
        descriptors = self.descr.create(atom, centers=positions)
        return self.pca.transform(descriptors)

def analyse_file(fpath, mode):
    n_particles = 1577
    n_rhod = 15
    if (mode=="soap_sort") or (mode=="lmbtr_sort"):
        
        # First figure out whether or not to use pd identifiers
        if not ("mcmd" in fpath):
            struct_path = os.path.abspath("src/localstructures_final_mc")
        else:
            pd_re = re.compile(r".*\/pd\/.*")
            if pd_re.match(fpath):
                print("Using PD structures")
                struct_path = os.path.abspath("src/localstructures_final_mcmd_pd")
            else:
                struct_path = os.path.abspath("src/localstructures_final_mcmd_rh")

        if mode == "soap_sort":
            r_cut=5.2 # 4.2 # 2.7 
            n_max=4
            l_max=3
            sigma=1.
            gamma_kernel=1.

            sorter = sort_neigh.NeighbourSort(
                local_structures_path=struct_path,
                r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma, gamma_kernel=gamma_kernel
            )
            fname = os.path.basename(fpath).split('.')[0]
            save_txt_path = os.path.join(os.path.dirname(fpath), fname+"_soap_sorted_counts.txt")

        elif mode=="lmbtr_sort":
            n_spec = 180

            descr = LMBTR(
                species=["Rh", "Cu"],
            #    k2={
            #        "geometry": {"function": "distance"},
            #        "grid": {"min": 0, "max": 5, "n": 100, "sigma": 0.1},
            #        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
            #    },
                k3={
                    "geometry": {"function": "angle"},
                    "grid": {"min": 0, "max": 180, "n": n_spec, "sigma": 2.8},
                    "weighting": {"function": "unity"},
                },
                periodic=False,
                sparse=False,
                normalization="none",
                flatten=True
            )
            sorter = sort_neigh.NeighbourSort(
                local_structures_path=struct_path,
                descr_func=descr
            )
            fname = os.path.basename(fpath).split('.')[0]
            save_txt_path = os.path.join(os.path.dirname(fpath), fname+"_lmbtr_sorted_counts.txt")

        sorter.load_particle(fpath)
        cat_counter = sorter.create_local_structure(last_n=n_rhod, create_subfolders=False, mode="class_all", cutoff_mult=0.98)

        file_name = sorter.sort_save_cat(
            file_name=save_txt_path, cat_counter=cat_counter
        )

    elif (mode=="soap_gendescr") or (mode=="lmbtr_gendescr") or (mode=="lmbtrpca_gendescr"):
        trajectory = ase_read(fpath, index=':')
        n_timesteps = len(trajectory)
        ts_buffersize = 100 #How many timesteps to save in numpy array before flushing to file

        for n_particle in range(n_timesteps):
            trajectory[n_particle] = ase_sort(trajectory[n_particle]) # So Rh is always in the last indices
        
        particle_range = (len(trajectory[0]) - np.arange(n_rhod)[::-1])-1 # Last n_rhod atoms

        cut_off = natural_cutoffs(trajectory[0], mult=0.98)
        neighbour_list = NeighborList(cut_off, bothways=True, self_interaction=False)

        if mode=="soap_gendescr":
            r_cut=5.2
            n_max=4
            l_max=3
            sigma=1.
            gamma_kernel=1.

            descr = SOAP(species=["Rh", "Cu"], r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma, periodic=False)
            n_descr = descr.get_number_of_features()
            fname = os.path.basename(fpath).split('.')[0]
            save_txt_path = os.path.join(os.path.dirname(fpath), fname+"_soap_%ux%u.txt"%(n_timesteps, n_rhod))
        elif mode=="lmbtr_gendescr":
            n_spec = 180

            descr = LMBTR(
                species=["Rh", "Cu"],
            #    k2={
            #        "geometry": {"function": "distance"},
            #        "grid": {"min": 0, "max": 5, "n": 100, "sigma": 0.1},
            #        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
            #    },
                k3={
                    "geometry": {"function": "angle"},
                    "grid": {"min": 0, "max": 180, "n": n_spec, "sigma": 2.8},
                    "weighting": {"function": "unity"},
                },
                periodic=False,
                sparse=False,
                normalization="none",
                flatten=True
            )
            n_descr = descr.get_number_of_features()
            fname = os.path.basename(fpath).split('.')[0]
            save_txt_path = os.path.join(os.path.dirname(fpath), fname+"_lmbtr_%ux%u.txt"%(n_timesteps, n_rhod))

        elif mode=="lmbtrpca_gendescr":
            n_spec = 180
            n_pca = 10

            lmbtr = LMBTR(
                species=["Rh", "Cu"],
            #    k2={
            #        "geometry": {"function": "distance"},
            #        "grid": {"min": 0, "max": 5, "n": 100, "sigma": 0.1},
            #        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
            #    },
                k3={
                    "geometry": {"function": "angle"},
                    "grid": {"min": 0, "max": 180, "n": n_spec, "sigma": 2.8},
                    "weighting": {"function": "unity"},
                },
                periodic=False,
                sparse=False,
                normalization="none",
                flatten=True
            )

            struct_path = os.path.abspath("src/localstructures_final_mc")
            standard_classifier = sort_neigh.NeighbourClassifier(
                local_structures_path=struct_path,
                non_class_max=14
            )
            standard_classifier.load_identifiers(descr_func=lmbtr)
            dict_lmbtr = _descriptors_from_classifier(standard_classifier)
            pca = PCA(n_components=n_pca)
            lmbtr_red = pca.fit_transform(dict_lmbtr)
            descr = _PCA_quick(descr=lmbtr, pca=pca)

            n_descr = n_pca
            fname = os.path.basename(fpath).split('.')[0]
            save_txt_path = os.path.join(os.path.dirname(fpath), fname+"_lmbtrsoap_%ux%u.txt"%(n_timesteps, n_rhod))
        
        with open(save_txt_path, 'wb') as f:
            f.close()

        descriptors = np.zeros((ts_buffersize, n_rhod, n_descr), dtype=np.float32)
        for ii_ts in range(n_timesteps):
            cur_particle = trajectory[ii_ts]
            neighbour_list.update(cur_particle)
            # Calculate descriptors for each Rh atom
            for jj_rhod, index in enumerate(particle_range):
                neighbour_indices, dists = neighbour_list.get_neighbors(index)
                neighbour_indices = np.append(np.array([index]), neighbour_indices, axis=0)
                neighbour_particle = cur_particle[neighbour_indices]
                descriptors[ii_ts%ts_buffersize, jj_rhod, :] = descr.create(neighbour_particle, centers=[0])
            # Every ts_buffersize write to file
            if (ii_ts+1)%ts_buffersize == 0:
                print("Writing at timestep %u"%ii_ts)
                with open(save_txt_path, 'ab') as f:
                    np.savetxt(f, descriptors.reshape((ts_buffersize*n_rhod, n_descr)))
                    f.close()
        
        # Write leftover timesteps that weren't written in loop
        ts_leftovers = n_timesteps%ts_buffersize
        if ts_leftovers != 0:
            with open(save_txt_path, 'ab') as f:
                np.savetxt(f, descriptors[:ts_leftovers, ...].reshape((ts_leftovers*n_rhod, n_descr)))
                f.close()

    else:
        raise ValueError("Mode %s not known. Available modes: soap_sort, lmbtr_sort, soap_gendescr, lmbtr_gendescr"%mode)

def slurm_analyse_file(mode):
    with open("target_files.txt", "r") as f:
        file_list = f.read().split('\n')[:-1]

    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    analyse_file(file_list[task_id], mode=mode)

if __name__ == '__main__':
    parser = ArgumentParser(
        description="Some input arguments can be added here"
    )
    parser.add_argument(
        "-m", "--mode",
        dest="mode", default='soap_sort', type=str,
        help="What type of analysis to run using slurm."
    )
    parsed_args = parser.parse_args()

    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        slurm_analyse_file(mode=parsed_args.mode)

    elif 'RUNALL' in os.environ:
        with open("target_files.txt", "r") as f:
            file_list = f.read().split('\n')[:-1]

        for path in file_list:
            analyse_file(path, mode=parsed_args.mode)

    else:
        print("ENV: ", os.environ)