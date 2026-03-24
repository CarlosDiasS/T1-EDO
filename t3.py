import numpy as np
import matplotlib.pyplot as plt

N = 50
t_inicio, t_fim = 0, 20
t_previsao = 30 
y0 = 0.1

t_i = np.linspace(t_inicio, t_fim, N)
h = t_i[1] - t_i[0] 

def solucao_analitica(t):
    return 1 / (1 + 9 * np.exp(-t))

np.random.seed(42)
y_i = solucao_analitica(t_i) + np.random.normal(0, 0.01, N)

A = np.column_stack([np.full(len(y_i)-1, h), h * y_i[:-1], h * (y_i[:-1]**2)])
r = np.diff(y_i)
a_hat, _, _, _ = np.linalg.lstsq(A, r, rcond=None)

t_futuro = np.arange(t_inicio, t_previsao + h, h)
y_predito = np.zeros(len(t_futuro))
y_predito[0] = y0

for i in range(len(t_futuro) - 1):
    # O modelo aprendido: dy/dt = a0 + a1*y + a2*y^2
    dy_dt = a_hat[0] + a_hat[1]*y_predito[i] + a_hat[2]*(y_predito[i]**2)
    y_predito[i+1] = y_predito[i] + h * dy_dt

plt.figure(figsize=(12, 6))

plt.scatter(t_i, y_i, color='red', alpha=0.3, label='Dados de Treino (Ruidosos)')

t_real_longo = np.linspace(t_inicio, t_previsao, 300)
plt.plot(t_real_longo, solucao_analitica(t_real_longo), 'g-', alpha=0.5, label='Teoria (Alvo)')

plt.plot(t_futuro, y_predito, 'b--', linewidth=2, label='Predicao do Modelo Aprendido')

plt.axvline(x=t_fim, color='black', linestyle=':', label='Fim dos Dados de Treino')

plt.title('Validacao de Longo Prazo: O Modelo Aprendido consegue prever o futuro?')
plt.xlabel('Tempo (t)')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()
