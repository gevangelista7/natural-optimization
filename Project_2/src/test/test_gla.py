
import torch as t
from utils import generate_point_cloud_with_optimum, plot_points_da
from GeneticAlgorithm import GeneralizedLloydAlgorithm


if __name__ == "__main__":
    n_clusters = 5
    X, _, _, _ = generate_point_cloud_with_optimum(n_clusters=n_clusters,
                                                   core_points=50,
                                                   cores_dispersion=5,
                                                   intra_cluster_dispersion=1)
    X = t.tensor(X)
    gla = GeneralizedLloydAlgorithm(n_clusters=n_clusters,
                                    X=X)
    gla.run()

    plot_points_da(data_vectors=X.cpu(),
                   Y=gla.clusters[0],
                   title='best_ever_idv {}'.format(n_clusters))


