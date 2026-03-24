import numpy as np
import matplotlib.pyplot as plt

y0_real = 0.1
t_fim = 10
N = 50
t_i = np.linspace(0, t_fim, N)

def solucao_analitica(t):
    return 1 / (1 + 9 * np.exp(-t))

ruido = np.random.normal(0, 0.02, N)
y_i = solucao_analitica(t_i) + ruido

dt = t_i[1] - t_i[0]
dy_dt = np.diff(y_i) / dt

y_ajuste = y_i[:-1]
X = (y_ajuste * (1 - y_ajuste)).reshape(-1, 1) # Matriz de design
Z = dy_dt.reshape(-1, 1)

r_estimado, residuos, rank, s = np.linalg.lstsq(X, Z, rcond=None)
r_final = r_estimado[0][0]

print(f"Taxa r estimada: {r_final:.4f} (Valor real: 1.0)")

h = 0.2
t_euler = np.arange(0, t_fim + h, h)
y_est = np.zeros(len(t_euler))
y_est[0] = y_i[0] 

for i in range(len(t_euler) - 1):
    f_y = r_final * y_est[i] * (1 - y_est[i])
    y_est[i+1] = y_est[i] + h * f_y

plt.figure(figsize=(10, 6))
plt.scatter(t_i, y_i, color='red', alpha=0.4, label='Dados com Ruido')
plt.plot(t_euler, y_est, 'b-', linewidth=2, label=f'Ajuste lstsq (r={r_final:.3f})')
plt.plot(t_i, solucao_analitica(t_i), 'g--', alpha=0.6, label='Curva Teorica Real')

plt.title('Ajuste de Parametros via numpy.linalg.lstsq')
plt.xlabel('Tempo')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()