import numpy as np
from math import modf


def generate_rastrigin(A):
    # f(x) = A*n + sum(x_i - A*cos(2*pi*x_i))
    def rastrigin(X):
        res = A*len(X)
        for x in X:
            res += x**2 - A*np.cos(2*np.pi*x)
        return res
    return rastrigin


def mirror_rastrigin(x):
    for i in range(len(x)):
        if abs(x[i]) < 5.12:
            f, w = modf(x)
            x[i] = w - f


