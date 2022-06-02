import numpy.random as rd
import numpy as np
import torch
from .clustering_cost import clustering_cost


def generate_point_cloud(core_number, core_points, dimension=2, cores_dispersion=1, cluster_dispersion=1):
    cluster_centers = cores_dispersion * rd.normal(0, 1, [core_number, dimension])
    std_dev = 0
    dist_quad = 0
    data_vectors = []
    for center in cluster_centers:
        cluster = cluster_dispersion * rd.normal(0, 1, [core_points, dimension])
        std_dev += np.std(cluster)
        dist_quad += np.mean(cluster**2)
        data_vectors.append(cluster + center)

    data_vectors = np.concatenate(data_vectors)
    std_dev = std_dev / core_number
    dist_quad = dist_quad / core_number
    return data_vectors, dist_quad, std_dev


def generate_point_cloud_with_optimum(n_clusters, core_points, dimension=2, cores_dispersion=1,
                                      intra_cluster_dispersion=1, T=1):
    cluster_centers = cores_dispersion * rd.normal(0, 1, [n_clusters, dimension])
    data_vectors = []
    for center in cluster_centers:
        cluster = intra_cluster_dispersion * rd.normal(0, 1, [core_points, dimension])
        data_vectors.append(cluster + center)

    data_vectors = np.concatenate(data_vectors)
    minJ, minD = clustering_cost(data_vectors, cluster_centers, T=T)

    return data_vectors, minJ, minD, cluster_centers
