#!/usr/bin/env python3.9

import sys
sys.path.append("py_src/")

import sort_neigh as sn
import time
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

def main():
    neighbour_sort = sn.NeighbourSort()

    start = time.time()
    neighbour_sort.init_folder_structure("test_data/cunano.lammpstrj")
    part_count = neighbour_sort.create_local_structure(create_subfolders=False)
    stop = time.time()
    np.save("part_count.npy", part_count)
    print("Sorting took %u seconds."%(stop-start))

def analyze():
    part_count = np.load("part_count.npy")
    classifier = sn.NeighbourClassifier()
    
    classes = []
    particle_num = []
    for ii_id in range(part_count.shape[-1]):
        cat_string = classifier.id_to_cat(ii_id)
        classes.append(cat_string)

        num = int(cat_string.split('_', 1)[0])
        particle_num.append(num)

    sort_arr = np.argsort(np.asarray(particle_num))
    sort_arr = np.asarray(sort_arr, np.int32)
    x_plot = np.arange(part_count.shape[-1])
    sort_x_plot = x_plot[sort_arr]

    sorted_classes = []
    for sort_ind in sort_arr:
        sorted_classes.append(classes[sort_ind])
        
    sorted_part_count = np.take(part_count, sort_arr, axis=1)

    total_count = np.sum(part_count, axis=0)
    sort_total_count = np.sum(sorted_part_count, axis=0)
    
    sort_not_zero = sort_total_count != 0

    fig1, axis1 = plt.subplots(figsize=(12, 8))
    axis1.set_title('Categorized Surface Atoms over 100 Timesteps')
    axis1.bar(x_plot, height=total_count)
    axis1.set_xticks(x_plot, classes, rotation=320)
    axis1.set_yscale('log')
    axis1.grid(axis='y')

    fig, axis = plt.subplots(figsize=(12, 8))
    axis.set_title('Categorized Surface Atoms over 100 Timesteps')
    axis.bar(x_plot, height=sort_total_count)
    axis.set_xticks(x_plot, sorted_classes, rotation=320)
    axis.set_yscale('log')
    axis.grid(axis='y')

    fig3, axis3 = plt.subplots(figsize=(150, 7))
    im = axis3.imshow(sorted_part_count.T[:, :], aspect='auto', cmap="viridis")
    axis3.set_xlabel('timestamp')
    axis3.set_ylabel('category')
    axis3.set_yticks(x_plot, sorted_classes)
    ax_cb = fig3.colorbar(im)
    
    not_zero_classes = []
    for ii_nz, not_zero in enumerate(sort_not_zero):
        if not_zero:
            not_zero_classes.append(sorted_classes[ii_nz])
    x_plot_not_zero = np.arange(len(not_zero_classes))
    print(len(not_zero_classes), x_plot_not_zero.shape)

    fig2, axis2 = plt.subplots(figsize=(150, 7))
    im = axis2.imshow(sorted_part_count.T[sort_not_zero, :], aspect='auto', cmap="viridis")
    axis2.set_xlabel('timestamp')
    axis2.set_ylabel('category')
    axis2.set_yticks(x_plot_not_zero, not_zero_classes)
    ax_cb = fig2.colorbar(im)
    plt.show()

if __name__ == '__main__':
    main()
    analyze()