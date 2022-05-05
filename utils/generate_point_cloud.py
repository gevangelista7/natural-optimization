import numpy.random as rd
import numpy as np


def generate_point_cloud(core_number, core_points, dimension=2, cores_dispersion=1):
    rd.seed(1)

    cluster_centers = cores_dispersion * rd.normal(0, 1, [core_number, dimension])
    data_vectors = []
    for center in cluster_centers:
        data_vectors.append(center)
        for _ in range(core_points - 1):
            data_vectors.append(center + rd.normal(0, 1, dimension))
    data_vectors = np.array(data_vectors)

    return data_vectors
