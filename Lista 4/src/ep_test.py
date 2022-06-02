
import torch
from GeneticAlgorithm import EvolutionaryProgramming, GAEvolutionPlot
from OptimTestFunctions import ackley_t_inv


class FitnessFunction:
    def __init__(self, func):
        self.counter = 0
        self.function = func

    def evaluate(self, X):
        self.counter += 1
        return self.function(X)


if __name__ == "__main__":
    torch.no_grad()
    ackley_fitness = FitnessFunction(ackley_t_inv)
    EP = EvolutionaryProgramming(individual_dimension=20,
                                 fitness_function=ackley_fitness,
                                 tgt_fitness=-5e-2,
                                 max_ite=1e3,
                                 _eps0=1e-3,
                                 _lambda=600,
                                 _mu=300,
                                 filename='teste')

    result = EP.run()
    GAEvolutionPlot(EP.register.file_name).plot_evolution()

    print(result)

