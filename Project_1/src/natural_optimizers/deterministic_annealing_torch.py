import numpy as np
import numpy.random as rd
import torch

device = 'cuda' if torch.has_cuda else 'cpu'


def deterministic_annealing_torch(X, n_centroid, T0, Tmin, max_iterations, alpha, epsilon, delta):
    # convection: X -> np.array((number_of_data_points, dimension))
    dim = X.shape[-1]
    Y = torch.tensor(rd.normal(0, 1, size=(n_centroid, dim)), device=device, dtype=torch.float64)

    d_xy = torch.zeros((len(Y), len(X)), device=device, dtype=torch.float64)
    p_yx = torch.zeros((len(Y), len(X)), device=device, dtype=torch.float64)

    # initialize histories
    history_J = torch.zeros(max_iterations, device=device, dtype=torch.float64)
    history_J[0] = torch.inf

    history_T = torch.zeros(max_iterations, device=device, dtype=torch.float64)
    history_T[0] = T0

    history_D = torch.zeros(max_iterations, device=device, dtype=torch.float64)
    history_D[0] = torch.inf

    history_Y = torch.zeros((max_iterations, *Y.shape), device=device, dtype=torch.float64)
    history_Y[0] = Y

    finished = False
    T = T0
    i = 1

    while not finished:
        # Partition condition
        # for i_x in range(len(X)):
        for i_y in range(len(Y)):
            # d_xy[i_y, i_x] = torch.sum((X[i_x] - Y[i_y])**2)
            d_xy[i_y] = torch.sum((X - Y[i_y])**2, axis=1)

        p_yx = torch.exp(-d_xy/T)
        Z_x = torch.sum(p_yx, axis=0)
        p_yx = p_yx / Z_x

        # Centroid condition
        for i_y in range(len(Y)):
            Y[i_y, 0] = torch.dot(p_yx[i_y], X[:, 0]) / torch.sum(p_yx[i_y])
            Y[i_y, 1] = torch.dot(p_yx[i_y], X[:, 1]) / torch.sum(p_yx[i_y])

        # Cost Function and history
        history_J[i] = -T/len(X)*torch.sum(torch.log(Z_x))
        history_D[i] = torch.mean(torch.sum(p_yx * d_xy, axis=0))
        history_T[i] = T

        # Loop control
        if abs(history_J[i] - history_J[i-1])/abs(history_J[i-1]) < delta:
            T = alpha * T
            Y = Y + epsilon * torch.randn(Y.shape, device=device)

        history_Y[i] = Y
        i += 1
        if (T < Tmin) or (i == max_iterations):
            finished = True

    return Y, p_yx, i, history_J, history_D, history_T, history_Y


