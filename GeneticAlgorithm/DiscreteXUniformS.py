
import torch as t
import numpy.random as rd
from Recombination import Recombination


class DiscreteXUniformS(Recombination):
    def __init__(self, offspring_tensor, parents_tensor, alpha=0.5, device='cpu'):
        super().__init__(offspring_tensor, parents_tensor)
        self.alpha = alpha
        self.x_values_p = self.parents_tensor[:, :self.dimension]
        self.sigma_p = self.parents_tensor[:, self.dimension:]
        self.device = device

    def parent_selector(self):
        return rd.choice(self.n_parent, 2, replace=False)

    def execute(self):
        for idx_child in range(self.offspring_tensor.shape[0]):
            idx_p0, idx_p1 = self.parent_selector()
            x_genes_pos0 = (t.rand(self.dimension, device=self.device) < .5).float()
            x_genes_pos1 = t.ones(self.dimension, device=self.device) - x_genes_pos0

            child_x = x_genes_pos0 * self.x_values_p[idx_p0] + x_genes_pos1 * self.x_values_p[idx_p1]
            child_s = self.alpha * self.sigma_p[idx_p0] + (1 - self.alpha) * self.sigma_p[idx_p1]

            self.offspring_tensor[idx_child] = t.concat((child_x, child_s))


