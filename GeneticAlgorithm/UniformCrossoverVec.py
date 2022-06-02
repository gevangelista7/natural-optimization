import numpy.random as rd
import numpy as np
import torch as t


class UniformCrossoverVec:
    def __init__(self, parents, offspring, dimensions):
        self.parents = parents
        self.offspring = offspring
        self.n_parents = len(self.parents)
        self.n_child = len(self.offspring)
        self.idx_p = np.arange(self.n_parents)
        self.dimensions = dimensions

    def execute(self):
        cut = int(self.n_child/2)
        for idx_c0, idx_c1 in zip(range(0, cut), range(cut, 2*cut)):
            idx_p0, idx_p1 = rd.choice(self.n_parents, 2, replace=False)
            self.offspring[idx_c0].copy_(self.parents[idx_p0])
            self.offspring[idx_c1].copy_(self.parents[idx_p1])
            crossed_genes = rd.uniform(0, 1, self.dimensions) > .5
            for idx_gene in range(self.dimensions):
                if crossed_genes[idx_gene]:
                    self.offspring[idx_c0][idx_gene] = self.parents[idx_p1][idx_gene]
                    self.offspring[idx_c1][idx_gene] = self.parents[idx_p0][idx_gene]





