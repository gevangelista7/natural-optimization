import torch as t
import numpy.random as rd
import numpy as np


class MutationMCL:
    """ Mutation type (\mu , \lambda) (mu comma lambda)"""

    def __init__(self, population, _tau1, _tau2, _eps0, x_lim=(-30, 30), device='cpu'):
        t.no_grad()
        self.population = population
        self.n_indiv, self.dimension = population.shape
        self.dimension = int((self.dimension / 2))
        self.population_x = population[:, :self.dimension]
        self.population_sigma = population[:, self.dimension:]
        self.x_lim = x_lim

        self._tau1 = _tau1
        self._tau2 = _tau2
        self.x_normal_std = 1
        self._eps0 = _eps0

        self.device = device

    def execute(self):
        self.population_sigma *= t.exp(self._tau1 * t.randn((1,), device=self.device)) + \
                                       self._tau2 * t.randn((self.n_indiv, self.dimension), device=self.device)
        self.population_sigma.clamp_(min=self._eps0)

        self.population_x += self.population_sigma * t.normal(0, self.x_normal_std, self.population_x.shape, device=self.device)
        self.population_x.clamp_(min=self.x_lim[0], max=self.x_lim[1])





