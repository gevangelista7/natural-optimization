from GeneticAlgorithm import FitnessFunctionWithCounter
import torch as t

# n_cluster :      Number of clusters
# n_points:        Number of points per cluster
# n_pop:           Number of individuals in population
# dim:             Number of dimensions

# Y.shape:          ( n_cluster , dim )   >> genotype
# popY.shape:       ( n_pop, n_cluster, dim )
# X.shape:          ( n_cluster * n_points, dim )

# idv.shape:        ( 1, n_cluster * n_dim )
# population.shape: ( n_pop, n_cluster * n_dim * 2 )

t.no_grad()


class ClusteringFitness(FitnessFunctionWithCounter):
    def __init__(self, X, n_clusters, popY=[], offspring_fitness=None, T=1, device='cuda'):
        super().__init__()
        self.X = X
        self.popY = popY
        self.offspring_fitness = offspring_fitness

        self.device = device
        self.T = T

        self.n_x, self.dim = X.shape
        self.n_y = n_clusters
        self.n_pop = len(popY)
        self.d_xy = t.empty((self.n_y, self.n_x), dtype=t.float64, device=self.device)

    def fitness_update(self):
        super().fitness_update()

        dx = self.X.expand(self.n_y, self.n_x, self.dim)
        dy = self.popY.expand(self.n_x, self.n_pop, self.n_y, self.dim).transpose(0, 1).transpose(1, 2)

        d_xy = t.sum((dy - dx) ** 2, dim=-1)

        p_yx = t.exp(- d_xy / self.T)
        p_yx.clamp_min_(1e-45)

        Z_x = t.sum(p_yx, dim=1)

        J = - self.T / self.n_x * t.sum(t.log(Z_x), dim=-1)
        self.fitness_array.copy_(-J.unsqueeze(1))

    def link(self, evaluation_population: t.Tensor, fitness_array: t.Tensor):
        super().link(evaluation_population=evaluation_population,
                     fitness_array=fitness_array)
        self.n_pop = evaluation_population.shape[0]
        self.decode()

    def decode(self):
        self.popY = self.evaluation_population.view(self.n_pop, self.n_y, self.dim)
        return self.popY

    def decode_idv(self, idv):
        idv = t.tensor(idv)[:self.n_y * self.dim]
        return idv.view(self.n_y, self.dim).cpu()
