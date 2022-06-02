import torch as t
import numpy.random as rd
import numpy as np


class BinaryMutationInt:
    def __init__(self, population, n_bit, pop_rate, dimension):
        self.population = population
        self.n_population = len(population)
        self.dimension = dimension
        self.n_mutants = int(self.n_population * pop_rate)
        self.n_bit = n_bit

    def execute(self):
        pop_idx = rd.choice(self.n_population, self.n_mutants, replace=False)
        mask = np.sum(2**rd.randint(self.dimension, size=(self.n_mutants, self.n_bit)), axis=1)
        self.population[pop_idx] ^= mask



