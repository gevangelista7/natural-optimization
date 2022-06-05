import torch as t


class SigmaControlRechemberg:
    def __init__(self, mutation, fitness_array, period=1e2, c=.99):
        self.mutation = mutation
        self.fitness_array = fitness_array
        self.c = c
        self.period = period

        self.last_best_fitness = t.max(fitness_array)

    def update_sigma(self, gen_n):
        if gen_n % self.period == 0:
            ps = sum((self.fitness_array > self.last_best_fitness).int())/len(self.fitness_array)
            self.last_best_fitness = t.max(self.fitness_array)
            if ps > .2:
                self.mutation.x_normal_std /= self.c
            elif ps < .2:
                self.mutation.x_normal_std *= self.c


class SigmaControlIterN:
    def __init__(self, mutation, final_iteration=2e3, sigma_ini=1, sigma_final=0.1):
        self.mutation = mutation
        self.final_iteration = final_iteration
        self.sigma_ini = sigma_ini
        self.sigma_final = sigma_final

    def update_sigma(self, gen_n):
        alpha = min(gen_n, self.final_iteration)/self.final_iteration
        sigma_val = alpha * self.sigma_final + (1 - alpha) * self.sigma_ini
        self.mutation.x_normal_std = sigma_val



