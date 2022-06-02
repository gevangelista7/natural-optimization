
import torch

def clustering_cost(data_vector, centroids, T=1):
    X = torch.tensor(data_vector)
    Y = torch.tensor(centroids)
    d_xy = torch.zeros((len(Y), len(X)), dtype=torch.float64)
    for i_y in range(len(Y)):
        d_xy[i_y] = torch.sum((X - Y[i_y]) ** 2, axis=1)

    p_yx = torch.exp(-d_xy / T)
    Z_x = torch.sum(p_yx, axis=0)
    J = - T / len(X) * torch.sum(torch.log(Z_x))
    D = torch.mean(torch.sum(p_yx * d_xy, axis=0))

    return J, D

