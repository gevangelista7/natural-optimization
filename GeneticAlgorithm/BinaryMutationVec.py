import torch as t
import numpy.random as rd
import numpy as np


class BinaryMutationVec:
    def __init__(self, population, n_bit, pop_rate):
        self.population = population
        self.n_population, self.dimension = population.shape
        self.n_mutants = int(self.n_population * pop_rate)
        self.n_bit = n_bit

    def execute(self):
        pop_idx = rd.choice(self.n_population, self.n_mutants, replace=False)
        mutation_idx = np.empty((self.n_mutants, self.n_bit), dtype=np.int)
        for i in range(len(mutation_idx)):
            mutation_idx[i] = rd.choice(self.dimension, self.n_bit, replace=False)

        for i in range(len(pop_idx)):
            g = mutation_idx[i]
            self.population[i, g] = ~self.population[i, g]



