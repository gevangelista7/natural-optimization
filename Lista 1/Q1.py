import numpy as np

# Question 1 - Integration by Monte Carlo Algorithm


if __name__ == "__main__":
    print("===============Início do item a===============")
    N = 1000000
    x = np.random.uniform(0, 1, N)
    res1 = np.mean(x ** 3)
    print('Resultado pelo Método de Monte Carlo com N = {}: \n{:.8f}'
          '\n---\n\n'.format(N, res1))

    randX = [0.9501, 0.2311, 0.6068]
    F_sum = 0
    i = 0
    for x in randX:
        F_x = x**3
        F_sum += F_x
        i += 1
        print('Passo: {} \nF_x: {:.4f} \nSoma acumulada: {:.4f}\n---'.format(i, F_x, F_sum))

    res2 = F_sum / len(randX)
    print("Resultado final (média a partir da soma acumulada e do tamanho da amostra): \n"
          "{:.4f}".format(res2))

    print("===============Fim do item a===============")
    print("===============Início do item b===============")
    N = 1000000
    x = np.random.uniform(0, 1, N)
    res3 = np.mean(x ** 2 * np.exp(-x))
    print('Resultado pelo Método de Monte Carlo (dist. uniforme) '
          'com N = {}: \n{:.8f}'
          '\n---\n\n'.format(N, res3))

    x = np.random.exponential(1, N)
    res4 = np.sum(x[x < 1]**2/N)
    print('Resultado pelo Método de Monte Carlo (dist. exponencial) '
          'com N = {}: \n{:.8f}'
          '\n---\n\n'.format(N, res4))

    randXexp = [0.0512, 1.4647, 0.4995, 0.7216]
    # randNexp = np.random.exponential(1, N)
    F_sum = 0
    i = 0
    for x in randXexp:
        if x >= 1:
            print("x > 1 : F_x não somado, passo não computado"
                  "\n---")
            continue
        F_x = x**2
        F_sum += F_x
        i += 1
        print('Passo: {} \nF_x: {:.4f} \nSoma acumulada: {:.4f}\n---'.format(i, F_x, F_sum))

    res5 = F_sum / len(randXexp)
    print("Resultado final (média a partir da soma acumulada e do tamanho da amostra): \n"
          "{:.4f}".format(res5))
    print("===============Fim do item b===============")

