def styblinski_tang(X):
    res = 0
    for x in X:
        res += x**4 - 16*x**2 + 5*x
    return res / 2
