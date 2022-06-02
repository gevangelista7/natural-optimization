import numpy.random as rd
import numpy as np


class UniformCrossoverInt:
    def __init__(self, parents, offspring, dimensions):
        self.parents = parents
        self.offspring = offspring
        self.n_parents = len(self.parents)
        self.n_child = len(self.offspring)
        self.idx_p = np.arange(self.n_parents)
        self.dimensions = dimensions

    def execute(self):
        if self.n_child % 2:
            self.offspring[0] = self.parents[rd.choice(self.n_parents)]
            ini = 1
        else:
            ini = 0

        for child0, child1 in self.offspring[ini:].view((-1, 2)):
            idx_p0, idx_p1 = rd.choice(self.n_parents, 2, replace=False)
            child0.copy_(self.parents[idx_p0])
            child1.copy_(self.parents[idx_p1])
            crossed_genes = rd.uniform(0, 1, self.dimensions) > .5

            for idx_gene in range(self.dimensions):
                if crossed_genes[idx_gene]:
                    mask = 2**idx_gene
                    child0.copy_(self.parents[idx_p1] & mask | child0 & (~mask))
                    child1.copy_(self.parents[idx_p0] & mask | child1 & (~mask))





