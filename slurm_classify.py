import os
import re
from argparse import ArgumentParser
from dscribe.descriptors import LMBTR, SOAP
from ase.io import read as ase_read
from ase.build import sort as ase_sort
from ase.neighborlist import natural_cutoffs, NeighborList
import numpy as np
import py_src.sort_neigh as sort_neigh

def analyse_file(fpath, mode):
    n_particles = 1577
    n_rhod = 15
    if (mode=="soap_sort") or (mode=="lmbtr_sort"):
        
        # First figure out whether or not to use pd identifiers
        pd_re = re.compile(r".*\/pd\/.*")
        if pd_re.match(fpath):
            struct_path = os.path.abspath("src/localstructures_newopt_pd")
        else:
            struct_path = os.path.abspath("src/localstructures_newopt_rh")

        if mode == "soap_sort":
            rcut=4.2
            nmax=4
            lmax=3
            sigma=0.6
            gamma_kernel=1.

            sorter = sort_neigh.NeighbourSort(
                local_structures_path=struct_path,
                rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma, gamma_kernel=gamma_kernel
            )
            fname = os.path.basename(fpath).split('.')[0]
            save_txt_path = os.path.join(os.path.dirname(fpath), fname+"_soap_sorted_counts.txt")

        elif mode=="lmbtr_sort":
            n_spec = 500

            lmbtr = LMBTR(
                species=["Rh", "Cu"],
            #    k2={
            #        "geometry": {"function": "distance"},
            #        "grid": {"min": 0, "max": 5, "n": 100, "sigma": 0.1},
            #        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
            #    },
                k3={
                    "geometry": {"function": "angle"},
                    "grid": {"min": 0, "max": 180, "n": n_spec, "sigma": 0.01},
                    "weighting": {"function": "unity"},
                },
                periodic=False,
                normalization="none",
                flatten=True
            )
            sorter = sort_neigh.NeighbourSort(
                local_structures_path=struct_path,
                descr_func=lmbtr
            )
            fname = os.path.basename(fpath).split('.')[0]
            save_txt_path = os.path.join(os.path.dirname(fpath), fname+"_lmbtr_sorted_counts.txt")

        sorter.load_particle(fpath)
        cat_counter = sorter.create_local_structure(last_n=n_rhod, create_subfolders=False, mode="class_all", cutoff_mult=0.98)

        file_name = sorter.sort_save_cat(
            file_name=save_txt_path, cat_counter=cat_counter
        )

    elif (mode=="soap_gendescr") or (mode=="lmbtr_gendescr"):
        trajectory = ase_read(fpath, index=':')
        n_timesteps = len(trajectory)

        for n_particle in range(n_timesteps):
            trajectory[n_particle] = ase_sort(trajectory[n_particle]) # So Rh is always in the last indices
        
        particle_range = (len(trajectory[0]) - np.arange(n_rhod)[::-1])-1 # Last n_rhod atoms

        cut_off = natural_cutoffs(trajectory[0], mult=0.98)
        neighbour_list = NeighborList(cut_off, bothways=True, self_interaction=False)

        if mode=="soap_gendescr":
            rcut=4.2
            nmax=4
            lmax=3
            sigma=0.6
            gamma_kernel=1.

            descr = SOAP(species=["Rh", "Cu"], rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma, periodic=False)
            n_descr = descr.get_number_of_features()
            fname = os.path.basename(fpath).split('.')[0]
            save_txt_path = os.path.join(os.path.dirname(fpath), fname+"_soap.npy")

        elif mode=="lmbtr_gendescr":
            n_spec = 500

            descr = LMBTR(
                species=["Rh", "Cu"],
            #    k2={
            #        "geometry": {"function": "distance"},
            #        "grid": {"min": 0, "max": 5, "n": 100, "sigma": 0.1},
            #        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
            #    },
                k3={
                    "geometry": {"function": "angle"},
                    "grid": {"min": 0, "max": 180, "n": n_spec, "sigma": 0.01},
                    "weighting": {"function": "unity"},
                },
                periodic=False,
                normalization="none",
                flatten=True
            )
            n_descr = descr.get_number_of_features()
            fname = os.path.basename(fpath).split('.')[0]
            save_txt_path = os.path.join(os.path.dirname(fpath), fname+"_lmbtr.npy")
        
        descriptors = np.zeros((n_timesteps, n_rhod, n_descr), dtype=np.float32)
        for ii_ts in range(n_timesteps):
            cur_particle = trajectory[ii_ts]
            neighbour_list.update(cur_particle)
            for jj_rhod, index in enumerate(particle_range):
                neighbour_indices, dists = neighbour_list.get_neighbors(index)
                neighbour_indices = np.append(np.array([index]), neighbour_indices, axis=0)
                neighbour_particle = cur_particle[neighbour_indices]
                descriptors[ii_ts, jj_rhod, :] = descr.create(neighbour_particle, positions=[0])
            
        with open(save_txt_path, 'wb') as f:
            np.save(f, descriptors)
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