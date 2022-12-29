import os
import py_src.sort_neigh as sort_neigh

def analyse_file(fpath):
    n_particles = 1577
    n_rhod = 15

    rcut=4.2
    nmax=4
    lmax=3
    sigma=0.6
    gamma_kernel=1.

    sorter = sort_neigh.NeighbourSort(
        local_structures_path=os.path.abspath("src/localstructures_newopt_rh"),
        rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma, gamma_kernel=gamma_kernel
    )

    sorter.load_particle(fpath)
    cat_counter = sorter.create_local_structure(last_n=n_rhod, create_subfolders=False, mode="class_all")
    
    fname = os.path.basename(fpath).split('.')[0]
    save_txt_path = os.path.join(os.path.dirname(fpath), fname+"_sorted_counts.txt")

    file_name = sorter.sort_save_cat(
        file_name=save_txt_path, cat_counter=cat_counter
    )

def slurm_analyse_file():
    with open("target_files.txt", "r") as f:
        file_list = f.read().split('\n')[:-1]

    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    analyse_file(file_list[task_id])

if __name__ == '__main__':
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        slurm_analyse_file()

    elif 'RUNALL' in os.environ:
        with open("target_files.txt", "r") as f:
            file_list = f.read().split('\n')[:-1]

        for path in file_list:
            analyse_file(path)

    else:
        print("ENV: ", os.environ)