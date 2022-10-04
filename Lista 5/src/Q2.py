import numpy as np
import torch as t
from GeneticAlgorithm import StandardGeneticAlgorithmMultiObj2, FitnessFunctionWithCounter
import matplotlib.pyplot as plt


class FonsecaFlemingFitness(FitnessFunctionWithCounter):
    def __init__(self, n_dim, n_bit, n_pop, device):
        super(FonsecaFlemingFitness, self).__init__()
        # self.n_pop = n_pop
        self.n_dim = n_dim
        self.n_bit = n_bit
        self.device = device

        self.min_val = -4
        self.max_val = 4

        self.fitness_array = None
        self.evaluation_population = None

        self.bin_arrays = None
        self.decoded_values = None

        self.decode_mask = 2 ** t.arange(self.n_bit, device=device)

    def link(self, evaluation_population: t.Tensor, fitness_array: t.Tensor):
        super(FonsecaFlemingFitness, self).link(evaluation_population=evaluation_population,
                                                fitness_array=fitness_array)
        self.n_pop = evaluation_population.shape[0]
        self.bin_arrays = self.evaluation_population.view(self.n_pop, self.n_dim, self.n_bit)

    def decode_values(self):
        pow2 = self.bin_arrays * self.decode_mask
        pos = t.sum(pow2, dim=-1)
        self.decoded_values = self.min_val + (self.max_val - self.min_val) * pos / 2 ** self.n_bit


class FitnessF1(FonsecaFlemingFitness):
    def fitness_update(self):
        FitnessFunctionWithCounter.fitness_update(self)
        self.decode_values()
        self.fitness_array.copy_(-(1 - t.exp(-t.sum((self.decoded_values - 1 / np.sqrt(self.n_dim)) ** 2, dim=-1))))


class FitnessF2(FonsecaFlemingFitness):
    def fitness_update(self):
        FitnessFunctionWithCounter.fitness_update(self)
        self.decode_values()
        self.fitness_array.copy_(-(1 - t.exp(-t.sum((self.decoded_values + 1 / np.sqrt(self.n_dim)) ** 2, dim=-1))))


def func_f1(x):
    return 1 - t.exp(-t.sum((x - 1 / np.sqrt(x.shape[-1])) ** 2, dim=-1))


def func_f2(x):
    return 1 - t.exp(-t.sum((x + 1 / np.sqrt(x.shape[-1])) ** 2, dim=-1))


if __name__ == '__main__':
    t.no_grad()

    # 2*16 (dim, bits) bits code,
    # 16 bits for each x, 16 for each y value.
    # x_i in [-4, 4]
    # individual genome: 32 bits
    # genome map: [x1,y1,x2, y2, ... ]

    n_dim = 2
    n_bit = 16
    n_population = 32
    n_island = 64

    f1 = FitnessF1(n_dim=n_dim,
                   n_bit=n_bit,
                   n_pop=n_population,
                   device='cuda')

    f2 = FitnessF2(n_dim=n_dim,
                   n_bit=n_bit,
                   n_pop=n_population,
                   device='cuda')

    GA = StandardGeneticAlgorithmMultiObj2(dimension=n_dim * n_bit,
                                           n_population=n_population,

                                           mutation_rate=.5,
                                           n_bit_mutate=4,
                                           epoch=30,
                                           n_migrant=4,
                                           n_islands=n_island,

                                           fitness_function1=f1,
                                           fitness_function2=f2,

                                           max_ite=3000)
    pop_result = GA.run()
    val_f1 = t.empty(pop_result.shape[0])
    val_f2 = t.empty(pop_result.shape[0])

    f1.link(evaluation_population=pop_result, fitness_array=val_f1)
    f1.fitness_update()
    f2.link(evaluation_population=pop_result, fitness_array=val_f2)
    f2.fitness_update()



    plt.scatter(-val_f1, -val_f2, label='Resultado')
    plt.title(r"Fonseca-Fleming Function ($X \in R^2$)")
    plt.xlabel(r"$f_1(X)$")
    plt.ylabel(r"$f_2(X)$")

    x = t.linspace(-1, 1, 50).view(-1, 1)
    plt.plot(func_f1(x), func_f2(x), 'r--', label='Fronteira analítica')
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.grid()

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()
