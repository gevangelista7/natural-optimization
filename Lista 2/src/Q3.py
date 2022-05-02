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


# calcJ = generate_rosenbrock(1, 100)
calcJ = generate_rastrigin(10)
# calcJ = styblinski_tang

if __name__ == "__main__":
    rd.seed(0)
    x_n = rd.normal(size=10)
    x_min = x_n
    J_n = calcJ(x_n)
    J_min = J_n
    N = 10000
    K = 128
    k = 1
    epsilon = 5e-2

    T_0 = 64
    T = T_0

    n = 0
    finished = False
    history = []

    while not finished:
        n += 1

        x_hat = x_n + rd.normal(size=len(x_n)) * epsilon
        J_possible = calcJ(x_hat)

        _deltaJ = J_possible - J_n
        r = rd.uniform(0, 1)

        if r < np.exp(-_deltaJ / T):
            x_n = x_hat
            J_n = J_possible
            if J_n < J_min:
                J_min = J_n
                x_min = x_n

        history.append((x_min, J_min, J_n))
        if n % N == 0:
            k += 1
            x_n = rd.normal(size=10)
            T = T_0 / np.log2(1 + k)
            if k > K:
                finished = True

    print("X_min = ", x_min, "\nJ_min = ", J_min)
    fig, ax = plt.subplots(2)
    ax[0].plot([data[-1] for data in history])
    ax[1].plot([data[1] for data in history])


