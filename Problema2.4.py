import numpy as np
import matplotlib.pyplot as plt

h_vetor = [1, 5, 10]

k = 0.1799  


def sol_analitica(ti, tf, dt, h0):
    n = int((tf - ti) / dt)
    t = np.linspace(ti, tf, n + 1)
    h = (-(k * t) / 2 + np.sqrt(h0)) ** 2

    return t, h


def df(t, h):
    return -k * np.sqrt(h)


def euler(ti, tf, dt, h0):
    n = int((tf - ti) / dt)
    t = np.linspace(ti, tf, n + 1)

    h = np.zeros(n + 1)

    h[0] = h0
    op = 0

    for i in range(n):

        aux = df(t[i], h[i])
        h[i + 1] = h[i] + dt * aux

        op += 1

        # caso a derivada seja negativa
        if h[i + 1] < 0:
            h[i + 1] = 0

    return t, h


def main():

    for h0 in h_vetor:
        t, h_num = euler(0, 10, 0.01, h0)
        ta, h_ana = sol_analitica(0, 10, 0.01, h0)

        plt.plot(t, h_num, label=f"Euler h0={h0}")
        plt.plot(ta, h_ana, "--", label=f"Analítica h0={h0}")

        # adicionar plot das medidas reais

    plt.xlabel("tempo")
    plt.ylabel("altura")
    plt.legend()
    plt.grid()
    plt.show()


main()

# a medida que passo(dt) aumenta, o erro do metodo de euler aumenta, pois a aproximacao das derivadas se torna menos precisa
# numericamente, encerra assim que o liquido esvazia mas, analiticamente, continua apos h=0, ou seja, assume valores negativos logo, nao fisicos

# o erro de euler e linear, aumenta conforme o passo aumenta
