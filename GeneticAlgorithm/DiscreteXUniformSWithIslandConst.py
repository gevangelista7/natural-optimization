
import torch as t
import numpy.random as rd
from Recombination import Recombination


class DiscreteXUniformSWithIslandConst(Recombination):
    def __init__(self, offspring_tensor, parents_tensor, n_islands, alpha=0.5, device='cpu'):
        super().__init__(offspring_tensor, parents_tensor)
        self.alpha = alpha
        self.x_values_p = self.parents_tensor[:, :self.dimension]
        self.sigma_p = self.parents_tensor[:, self.dimension:]
        self.device = device
        self.n_islands = n_islands

    def execute(self):
        for island in range(self.n_islands):
            parents = self.parents_tensor[(self.parents_tensor[:, 0] == island).nonzero().squeeze(1)]
            child_idx_list = (self.offspring_tensor[:, 0] == island).nonzero().squeeze(1)

            parents_x = parents[:, 1:self.dimension+1]
            parents_s = parents[:, self.dimension+1:]

            if parents.shape[0] == 0:
                continue

            for idx_child in child_idx_list:
                if parents.shape[0] == 1:
                    self.offspring_tensor[idx_child].copy_(parents[0])
                else:
                    idx_p0, idx_p1 = rd.choice(parents.shape[0], 2, replace=False)
                    x_genes_pos0 = (t.rand(self.dimension, device=self.device) < .5).float()
                    x_genes_pos1 = t.ones(self.dimension, device=self.device) - x_genes_pos0

                    child_x = x_genes_pos0 * parents_x[idx_p0] + x_genes_pos1 * parents_x[idx_p1]
                    child_s = self.alpha * parents_s[idx_p0] + (1 - self.alpha) * parents_s[idx_p1]

                    self.offspring_tensor[idx_child].copy_(t.concat((t.tensor(island, device=self.device).unsqueeze(0),
                                                                 child_x,
                                                                 child_s)))
