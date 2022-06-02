from utils import generate_point_cloud_with_optimum
import torch as t
from GeneticAlgorithm import StandardGeneticAlgorithmVec, GAEvolutionPlot
from utils import plot_points_da


def clustering_cost(X, Y):
    d_xy = t.zeros((len(Y), len(X)), dtype=t.float64)
    T = 1
    for i_y in range(len(Y)):
        d_xy[i_y] = t.sum((X - Y[i_y]) ** 2, axis=1)

    p_yx = t.exp(-d_xy / T)
    Z_x = t.sum(p_yx, axis=0)
    J = - T / len(X) * t.sum(t.log(Z_x))

    return J


def clustering_cost_pop(X, popY):
    popJ = t.empty(len(popY))
    for i in range(len(popY)):
        popJ[i] = clustering_cost(X, popY[i])
    return popJ


def decode_val(bin_array, min_val, max_val):
    n_bit = bin_array.shape[-1]
    pow2 = bin_array * 2 ** t.arange(n_bit, device='cuda')
    pos = t.sum(pow2, axis=1)
    val = min_val + (max_val - min_val) * pos / 2 ** n_bit
    return val


def decode(genome, n_pop, n_centroid, n_dim, min_val, max_val):
    values = decode_val(genome.view((n_pop * n_centroid * n_dim, -1)), min_val=min_val, max_val=max_val)
    values = values.reshape((n_pop, n_centroid, n_dim))
    return values


def fitness_cluster_sga(genome, data_vector, minJ):
    values = decode_(genome)
    return minJ - clustering_cost_pop(data_vector, values)


if __name__ == '__main__':
    t.no_grad()

    # 10*2*16 (NC, dim, bits) bits code,
    # 16 bits for each x, 16 for each y value.
    # x in [-15, 15], y in [-15, 15]
    # resolution: 0.000457763671875
    # individual genome: 320 bits
    # genome map: [x1,y1,x2, y2, ... ]

    NC = 4
    CP = 100
    n_population = 50
    dimension = 320


    def decode_(genome):
        return decode(genome, n_pop=n_population, n_centroid=NC, n_dim=2, min_val=-15, max_val=15)


    data_vector, minJ, minD, _ = generate_point_cloud_with_optimum(n_clusters=NC, core_points=CP, cores_dispersion=10)
    data_vector = t.tensor(data_vector, device='cuda')

    def fitness_cluster_sga_(genome):
        return fitness_cluster_sga(genome, data_vector, minJ)

    SGA = StandardGeneticAlgorithmVec(dimension=dimension, n_population=n_population,
                                      fitness_function=fitness_cluster_sga_, mutation_rate=.85,
                                      tgt_fitness=-.15*minJ, n_bit_mutate=45, max_ite=1e3)

    result = SGA.run()
    GAEvolutionPlot(SGA.register.file_name).plot_evolution()

    best_idv = t.tensor(result['best_idv'], device='cuda')
    Y_decoded = decode(best_idv, n_pop=1, n_centroid=NC, n_dim=2, min_val=-15, max_val=15)
    plot_points_da(data_vector.cpu(), Y=Y_decoded[0].cpu(), with_voronoi=True, title="Best idv")
    plot_points_da(data_vector.cpu(), Y=Y_decoded[0].cpu(), with_voronoi=False, title="Best idv")

    for key in result.keys():
        print(key, " : ", result[key])

