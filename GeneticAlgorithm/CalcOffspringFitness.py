
class CalcOffspringFitness:
    def __init__(self, fitness_function, n_indiv):
        self.fitness_function = fitness_function
        self.n_indiv = n_indiv

    def calc_offspring_fitness(self, x_values, offspring_fitness):
        for i in range(self.n_indiv):
            offspring_fitness[i] = self.fitness_function.evaluate(x_values[i])

