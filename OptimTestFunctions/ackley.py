import numpy as np
from .generate_ackley import generate_ackley, generate_ackley_t

ackley = generate_ackley(20, 0.2, 2*np.pi)
ackley_t = generate_ackley_t(20, 0.2, 2*np.pi)


def ackley_t_inv(X):
    return - ackley_t(X)
