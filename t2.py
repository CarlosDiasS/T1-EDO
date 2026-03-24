
import numpy as np
import matplotlib.pyplot as plt

N = 50
t_inicio, t_fim = 0, 10
t_previsao = 15 
y0 = 0.1
h = 0.2

t_i = np.linspace(t_inicio, t_fim, N)
def solucao_analitica(t): return 1 / (1 + 9 * np.exp(-t))

np.random.seed(42)
y_i = solucao_analitica(t_i) + np.random.normal(0, 0.01, N)
dt = t_i[1] - t_i[0]
dy_dt = np.diff(y_i) / dt
y_ajuste = y_i[:-1]

X2 = np.column_stack([np.ones_like(y_ajuste), y_ajuste, y_ajuste**2])
a_grau2, _, _, _ = np.linalg.lstsq(X2, dy_dt, rcond=None)

X1 = np.column_stack([np.ones_like(y_ajuste), y_ajuste])
a_grau1, _, _, _ = np.linalg.lstsq(X1, dy_dt, rcond=None)

t_futuro = np.arange(t_inicio, t_previsao + h, h)
y_p2 = np.zeros(len(t_futuro)); y_p1 = np.zeros(len(t_futuro))
y_p2[0] = y_p1[0] = y0

for i in range(len(t_futuro) - 1):
    # Euler Grau 2
    y_p2[i+1] = y_p2[i] + h * (a_grau2[0] + a_grau2[1]*y_p2[i] + a_grau2[2]*y_p2[i]**2)
    # Euler Grau 1
    y_p1[i+1] = y_p1[i] + h * (a_grau1[0] + a_grau1[1]*y_p1[i])

plt.figure(figsize=(10, 6))
plt.scatter(t_i, y_i, color='red', alpha=0.3, label='Dados de Treino')
plt.plot(t_futuro, solucao_analitica(t_futuro), 'g-', label='Teoria Real (Alvo)')
plt.plot(t_futuro, y_p2, 'b--', linewidth=2, label='Modelo Grau 2 (Saturacao)')
plt.plot(t_futuro, y_p1, 'm:', linewidth=2, label='Modelo Grau 1 (Exponencial)')

plt.axvline(x=t_fim, color='k', linestyle='--', alpha=0.5)
plt.ylim(0, 2) # Limita o eixo Y para ver a explosao do Grau 1
plt.title('Comparacao: Grau 1 vs Grau 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()