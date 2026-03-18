import numpy as np
import matplotlib.pyplot as plt

# 1. Definir a solução analítica
def solucao_analitica(t):
    return 1 / (1 + 9 * np.exp(-t))

# 2. Gerar 50 pontos de 0 a 10  obs: aumentando para 20, percebemos melhor que ira ao infinito
t_i = np.linspace(0, 10, 50)
y_real = solucao_analitica(t_i)

# 3. Adicionar ruído gaussiano N(0, 0.01)
ruido = np.random.normal(0, 0.01, 50)
y_i = y_real + ruido

# 4. Plotar
plt.scatter(t_i, y_i, label='Dados Experimentais (com ruído)', color='red', s=10)
plt.plot(t_i, y_real, label='Solução Analítica', alpha=0.7)
plt.xlabel('Tempo (t)')
plt.ylabel('y(t)')
plt.legend()
plt.show()