import torch as t
from .GLAStep import GLAStep
from utils import plot_points_da, generate_point_cloud_with_optimum


class GeneralizedLloydAlgorithm:
    def __init__(self, X, n_clusters, pop0_dispersion=1):
        self.X = X
        self.dim = X.shape[-1]
        self.n_clusters = n_clusters
        self.clusters = t.normal(0, pop0_dispersion, (self.n_clusters, self.dim)).unsqueeze(0)
        self.old_clusters = t.empty(self.clusters.shape)
        self.gla_stepper = GLAStep(data_vector=self.X, original=self.old_clusters, neighbor=self.clusters, n_clusters=n_clusters)

    def run(self):
        while not self.clusters.isclose(self.old_clusters).all():
            self.old_clusters.copy_(self.clusters)
            self.gla_stepper.update_neighbors()








