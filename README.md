# Readme

Github repository for classifying the type of surface on CuRh nanoparticles.
See `py_test` for an introductory jupyter script in `show_sort_neigh.ipynb`.
The functions written for the main classification script in `slurm_classify.py` show full usage of all available classes in `py_src/sort_neigh.py`.

## Example of Usage

A basic process for using some available functionalities is given below.
```Python
import sort_neigh

# Define a sorter
sorter = sort_neigh.NeighbourSort(r_cut=3.1, n_max=4, l_max=3, sigma=0.5, gamma_kernel=0.05)

# Import a trajectory of a nanoparticle
nano_path = "sometrajectory.lammpstrj"
sorter.load_particle(
    nano_path, timesteps=100
)

# Classify sites of Rh
cat_counter = sorter.create_local_structure(
    last_n=14,
    mode="pre_group"
)

# Save to .txt file
file_name = sorter.sort_save_cat(
    file_name=save_txt_path, cat_counter=cat_counter
)
```