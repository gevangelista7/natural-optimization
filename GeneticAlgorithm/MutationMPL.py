import torch as t


class MutationMPL:
    """ Mutation type (\mu + \lambda) (mu plus lambda)"""

    def __init__(self, population, offspring, _tau1, _tau2, _eps0, x_lim=(-30, 30), rate=None, device='cpu'):
        t.no_grad()

        self.population = population
        self._mu, self.dimension = population.shape
        self.dimension = int((self.dimension / 2))

        self.offspring = offspring
        self._lambda = len(offspring)
        self.offspring_x = offspring[:, :self.dimension]
        self.offspring_sigma = offspring[:, self.dimension:]

        self._tau1 = _tau1
        self._tau2 = _tau2
        self.rate = rate
        self._eps0 = _eps0
        self.x_lim = x_lim

        self.device = device

    def execute(self):
        for child_idx in range(self._lambda):
            self.offspring[child_idx] = self.population[child_idx % self._mu]

        self.offspring_sigma[self._mu:] *= t.exp(self._tau1 * t.normal(0, 1, (1,), device=self.device) +
                                                 self._tau2 * t.normal(0, 1, (self._lambda-self._mu, self.dimension), device=self.device))
        self.offspring_sigma.clamp_(min=self._eps0)

        self.offspring_x.clamp_(min=self.x_lim[0], max=self.x_lim[1])
        self.offspring_x[self._mu:] += self.offspring_sigma[self._mu:] * t.normal(0, 1, (self._lambda-self._mu, self.dimension), device=self.device)

