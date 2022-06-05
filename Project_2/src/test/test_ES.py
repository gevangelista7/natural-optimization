import torch as t
from GeneticAlgorithm import EvolutionStrategy, GAEvolutionPlot
from ClusteringFitness import ClusteringFitness
from utils import generate_point_cloud_with_optimum, plot_points_da

# n_cluster :      Number of clusters
# n_points:        Number of points per cluster
# n_pop:           Number of individuals in population
# dim:             Number of dimensions

# Y.shape:          ( n_cluster , dim )   >> genotype
# popY.shape:       ( n_pop, n_cluster, dim )
# X.shape:          ( n_cluster * n_points, dim )

# idv.shape:        ( 1, n_cluster * n_dim )
# population.shape: ( n_pop, n_cluster * n_dim * 2 )

if __name__ == '__main__':
    t.set_grad_enabled(False)

    n_clusters = 16
    dim = 2
    X, minJ, minD, centers = generate_point_cloud_with_optimum(n_clusters=n_clusters,
                                                               core_points=100,
                                                               cores_dispersion=n_clusters,
                                                               dimension=dim,
                                                               T=1)
    X = t.tensor(X, device='cuda')
    X_limit = t.max(abs(X))
    fitness_function = ClusteringFitness(X=X,
                                         n_clusters=n_clusters,
                                         T=1)

    ES = EvolutionStrategy(individual_dimension=n_clusters*dim,
                           fitness_function=fitness_function,
                           tgt_fitness=-1.1*minJ,
                           max_eval=2e5,
                           _eps0=1e-3,
                           _lambda=500,
                           _mu=20,
                           _tau1=.8,
                           _tau2=.5,
                           filename='teste',
                           x_lim=(-X_limit, X_limit))

    result = ES.run()
    GAEvolutionPlot(ES.iter_register.complete_filename).plot_evolution()
    Ybest = fitness_function.decode_idv(result['best_idv']).cpu()

    plot_points_da(data_vectors=X.cpu(), Y=Ybest, title='best_ever_idv {}'.format(n_clusters), with_voronoi=True)
    # plot_points_da(data_vectors=X.cpu(), Y=Ybest, title='best_ever_idv {}'.format(n_clusters), with_voronoi=False)

    print(result)





