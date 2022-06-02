from math import modf


def generate_rosenbrock(a, b):
    # f(x,y) = (a - x)**2 + b*(y - x**2)**2 -> min global = (a, a**2)
    # f(x) = sum(b*(x_(i+1) - x_i**2) + (a - x_i)**2)
    def rosenbrock(X):
        res = 0
        for i in range(len(X)-1):
            res += b * (X[i+1] - X[i]**2)**2 + (a - X[i])**2
        return res
    return rosenbrock


def mirror_rosenbrock(x):
    for i in range(len(x)):
        if abs(x[i]) < 3:
            f, w = modf(x)
            x[i] = w - f
