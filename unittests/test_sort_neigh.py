from ase.io import read as ase_read
from ase.neighborlist import natural_cutoffs, NeighborList
import numpy as np
import os
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

    def test_ml_classifier(self):
        test_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")
        particle_file = os.path.abspath(os.path.join(test_data, "cunano_show.lammpstrj"))

        full_particle = ase_read(particle_file)
        
        n_clusters = 9
        ml_classifier = sn.USMLClassifier()
        train_clusters = ml_classifier.train_on_particle(full_particle, nmax=4, lmax=3, dim_red=PCA(n_components=2), clusterer=KMeans(n_clusters=n_clusters))
        predict_clusters = ml_classifier.classify(full_particle)
        
        assert train_clusters.shape == predict_clusters.shape
        assert np.all(train_clusters==predict_clusters)

        cut_off = natural_cutoffs(full_particle, mult=4)
        neighbour_list = NeighborList(cut_off, bothways=True, self_interaction=False)
        neighbour_list.update(full_particle)
        
        soaps = ml_classifier.soaper.create(full_particle)
        red_part = ml_classifier.dim_red.transform(soaps)
        # plt.scatter(red_part[:, 0], red_part[:, 1], marker='s', c=train_clusters)

        series_cluster = np.zeros((len(full_particle), ), dtype=np.int32)
        plot_2 = np.zeros((len(full_particle), 2))
        for atom_index in range(len(full_particle)):
            neighbour_indices, trash = neighbour_list.get_neighbors(atom_index)
            neighbour_indices = np.append(np.array([atom_index]), neighbour_indices, axis=0)

            neighbour_particle = full_particle[neighbour_indices]
            series_cluster[atom_index] = ml_classifier.classify(neighbour_particle)[0]

            soaps = ml_classifier.soaper.create(neighbour_particle)
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