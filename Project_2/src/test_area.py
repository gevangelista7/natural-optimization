import torch as t
from ClusteringFitness import ClusteringFitness
import unittest
from utils import generate_point_cloud_with_optimum


class ClusteringFitnessTests(unittest.TestCase):
    def test_functionality(self):
        dim = 2
        n_cores = 4
        n_pop = 2

        X, minJ, minD, centers = generate_point_cloud_with_optimum(n_clusters=n_cores, core_points=10, dimension=dim)
        X = t.tensor(X)

        popY = t.normal(0, 1, (n_pop, n_cores, dim))
        popY[0] = t.tensor(centers)

        fit = t.zeros(n_pop)

        fit_updater = ClusteringFitness(X=X, popY=popY, fitness_array=fit)

        fit_updater.clustering_cost_update()

        self.assertAlmostEqual(fit[0], minJ, delta=1e-6)
        self.assertNotAlmostEqual(fit[1], 0, delta=1e-6)
