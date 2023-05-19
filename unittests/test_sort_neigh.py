from ase.io import read as ase_read
from ase.neighborlist import natural_cutoffs, NeighborList
import numpy as np
import os
import shutil
import sys
import unittest

from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py_src/"))

import sort_neigh as sn

class TestSortNeigh(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")
        self.particle_file = os.path.abspath(os.path.join(self.test_data, "cunano_show.lammpstrj"))

    def test_ase_classification(self):
        dir_sorter = sn.NeighbourSort(r_cut=3.1, n_max=4, l_max=3, sigma=0.5, gamma_kernel=0.05)
        ram_sorter = sn.NeighbourSort(r_cut=3.1, n_max=4, l_max=3, sigma=0.5, gamma_kernel=0.05)

        target_folder = os.path.join(os.path.dirname(__file__), "res")

        dir_sorter.init_folder_structure(
            self.particle_file, n_atoms_in_part=1400, timesteps=100, out_dir=target_folder
        )
        dir_cat_counter = dir_sorter.create_local_structure(
            last_n=14, create_subfolders=False,
            mode="pre_group"
        )
        shutil.rmtree(target_folder)

        ram_sorter.load_particle(self.particle_file)
        ram_cat_counter = ram_sorter.create_local_structure(
            last_n=14,
            mode="pre_group"
        )

        guessed_differently = np.sum(dir_cat_counter != ram_cat_counter)
        assert guessed_differently == 0, "Counted differently in %u places."%guessed_differently



    def test_ml_classifier(self):
        full_particle = ase_read(self.particle_file)
        
        n_clusters = 9
        ml_classifier = sn.USMLClassifier()
        train_clusters = ml_classifier.train_on_particle(full_particle, n_max=4, l_max=3, dim_red=PCA(n_components=2), clusterer=KMeans(n_clusters=n_clusters))
        predict_clusters = ml_classifier.classify(full_particle)
        
        assert train_clusters.shape == predict_clusters.shape
        assert np.all(train_clusters==predict_clusters)

        cut_off = natural_cutoffs(full_particle, mult=4)
        neighbour_list = NeighborList(cut_off, bothways=True, self_interaction=False)
        neighbour_list.update(full_particle)
        
        soaps = ml_classifier.descr.create(full_particle)
        red_part = ml_classifier.dim_red.transform(soaps)
        # plt.scatter(red_part[:, 0], red_part[:, 1], marker='s', c=train_clusters)

        series_cluster = np.zeros((len(full_particle), ), dtype=np.int32)
        plot_2 = np.zeros((len(full_particle), 2))
        for atom_index in range(len(full_particle)):
            neighbour_indices, trash = neighbour_list.get_neighbors(atom_index)
            neighbour_indices = np.append(np.array([atom_index]), neighbour_indices, axis=0)

            neighbour_particle = full_particle[neighbour_indices]
            series_cluster[atom_index] = ml_classifier.classify(neighbour_particle)[0]

            soaps = ml_classifier.descr.create(neighbour_particle)
            red_part = ml_classifier.dim_red.transform(soaps)
            plot_2[atom_index, 0] = red_part[0, 0]
            plot_2[atom_index, 1] = red_part[0, 1]

        # plt.scatter(plot_2[:, 0], plot_2[:, 1], marker='o', c=series_cluster)
        # plt.show()
        ndiffs = np.sum(series_cluster != predict_clusters)
        
        # assert np.all(series_cluster==predict_clusters), "Found %u differences"%ndiffs

        for n_clust in range(n_clusters):
            sum_series = np.sum(series_cluster==n_clust)
            sum_predict = np.sum(predict_clusters==n_clust)
            assert sum_series == sum_predict, "Error found in %u cluster. Expected %u but predicted %u"%(n_clust, sum_predict, sum_series)

if __name__ == '__main__':
    unittest.main()