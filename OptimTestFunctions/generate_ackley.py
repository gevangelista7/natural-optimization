import numpy as np
import torch as t


def generate_ackley(a, b, c):
    # f(x,y) = (a - x)**2 + b*(y - x**2)**2 -> min global = (a, a**2)
    # f(x) = sum(b*(x_(i+1) - x_i**2) + (a - x_i)**2)
    def ackley(X):
        d = len(X)
        s1 = np.sum(X**2)
        s2 = np.sum(np.cos(c*X))
        return - a * np.exp(-b * np.sqrt(1/d*s1)) - np.exp(1/d * s2) + a + np.exp(1)
    return ackley


def generate_ackley_t(a, b, c):
    # f(x,y) = (a - x)**2 + b*(y - x**2)**2 -> min global = (a, a**2)
    # f(x) = sum(b*(x_(i+1) - x_i**2) + (a - x_i)**2)
    def ackley_t(X):
        d = len(X[0])
        s1 = t.sum(X**2, axis=1)
        s2 = t.sum(t.cos(c*X), axis=1)
        return a * (1 - t.exp(-b * t.sqrt(s1/d))) + t.exp(t.tensor(1)) - t.exp(s2/d)
    return ackley_t


