import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from math import modf
from copy import deepcopy


def generate_rosenbrock(a, b):
    # f(x,y) = (a - x)**2 + b*(y - x**2)**2 -> min global = (a, a**2)
    # f(x) = sum(b*(x_(i+1) - x_i**2) + (a - x_i)**2)
    def rosenbrock(X):
        res = 0
        for i in range(len(X)-1):
            res += b * (X[i+1] - X[i]**2)**2 + (a - X[i])**2
        return res
    return rosenbrock


def generate_rastrigin(A):
    # f(x) = A*n + sum(x_i - A*cos(2*pi*x_i))
    def rastrigin(X):
        res = A*len(X)
        for x in X:
            res += x**2 - A*np.cos(2*np.pi*x)
        return res
    return rastrigin


def styblinski_tang(X):
    res = 0
    for x in X:
        res += x**4 - 16*x**2 + 5*x
    return res / 2


def mirror_rosenbrock(x):
    for i in range(len(x)):
        if abs(x[i]) < 3:
            f, w = modf(x)
            x[i] = w - f


def mirror_rastrigin(x):
    for i in range(len(x)):
        if abs(x[i]) < 5.12:
            f, w = modf(x)
            x[i] = w - f