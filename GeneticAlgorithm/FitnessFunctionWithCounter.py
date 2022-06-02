import torch as t

class FitnessFunctionWithCounter:
    def __init__(self):
        self.counter = 0
        self.fitness_array = []
        self.evaluation_population = []

    def fitness_update(self):
        self.counter += len(self.evaluation_population)

    def decode(self):
        pass

    def link(self, evaluation_population: t.Tensor, fitness_array: t.Tensor):
        self.evaluation_population = evaluation_population
        self.fitness_array = fitness_array



