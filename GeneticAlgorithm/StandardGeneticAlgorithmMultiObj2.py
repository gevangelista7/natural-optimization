
import torch as t
from .BinaryMutationVec import BinaryMutationVec
from .UniformCrossoverVec import UniformCrossoverVec
from .StochasticUniversalSampling import StochasticUniversalSampling
from .FitnessFunctionWithCounter import FitnessFunctionWithCounter
from copy import deepcopy


class StandardGeneticAlgorithmMultiObj2:
    def __init__(self,
                 dimension,
                 n_population,
                 n_bit_mutate,
                 mutation_rate,
                 epoch,
                 n_islands,
                 n_migrant,
                 fitness_function1: FitnessFunctionWithCounter,
                 fitness_function2: FitnessFunctionWithCounter,
                 max_ite):

        t.no_grad()

        self.epoch = epoch
        self.n_migrant = n_migrant
        self.n_islands = n_islands
        self.gen_n = 0
        self.max_ite = max_ite

        self.population_l = []
        self.offspring_l = []
        self.offspring_fitness_l = []
        self.fitness_function_l = []
        self.mutations_l = []
        self.recombination_l = []
        self.recombination = []
        self.survivors_selection_l = []

        for i in range(n_islands):

            population = t.empty((n_population, dimension), dtype=t.bool, device='cuda')
            population.random_()

            self.population_l.append(population)

            offspring = t.empty((n_population, dimension), dtype=t.bool, device='cuda')
            offspring.random_()

            self.offspring_l.append(offspring)

            if i % 2 == 0:
                fitness_function = deepcopy(fitness_function1)
            else:
                fitness_function = deepcopy(fitness_function2)

            offspring_fitness = t.zeros(n_population)
            fitness_function.link(evaluation_population=offspring,
                                  fitness_array=offspring_fitness)

            self.offspring_fitness_l.append(offspring_fitness)
            self.fitness_function_l.append(fitness_function)

            mutation = BinaryMutationVec(population=population,
                                         pop_rate=mutation_rate,
                                         n_bit=n_bit_mutate)

            self.mutations_l.append(mutation)

            recombination = UniformCrossoverVec(offspring=offspring,
                                                dimensions=dimension,
                                                parents=population)

            self.recombination_l.append(recombination)

            survivors_selection = StochasticUniversalSampling(offspring_fitness=offspring_fitness,
                                                              selected=population,
                                                              candidates=offspring)
            self.survivors_selection_l.append(survivors_selection)

    def run(self):

        self.gen_n = 0
        while self.gen_n < self.max_ite:
            for i in range(self.n_islands):
                self.recombination_l[i].execute()
                self.mutations_l[i].execute()
                self.fitness_function_l[i].fitness_update()

            if self.gen_n % self.epoch == 0:
                for i in range(self.n_islands-1):
                    _, best1_idx = self.offspring_fitness_l[i].topk(self.n_migrant)
                    _, best2_idx = self.offspring_fitness_l[i+1].topk(self.n_migrant)

                    best1 = t.clone(self.offspring_l[i][best1_idx])

                    self.offspring_l[i][best1_idx] = t.clone(self.offspring_l[i+1][best2_idx])
                    self.offspring_l[i+1][best2_idx] = t.clone(best1)

            for i in range(self.n_islands):
                self.survivors_selection_l[i].execute()

            self.gen_n += 1

        return t.concat(self.population_l)

