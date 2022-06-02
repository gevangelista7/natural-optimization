import torch as t


class GLAStep:
    def __init__(self, data_vector, original, neighbor, n_dim):
        self.X = data_vector
        self.original = original.view(-1, n_dim)
        self.neighbor = neighbor.view(-1, n_dim)

        self.n_y = self.original.shape[0]
        self.n_x = self.X.shape[0]
        self.n_dim = n_dim

        self.dx = data_vector.expand(self.n_y, self.n_x, self.n_dim)
        self.dy = t.empty(self.dx.shape)

    def update_neighbors(self):
        self.dy = self.original.expand(self.dx.shape)




