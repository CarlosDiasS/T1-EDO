import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAÇÕES DO PROBLEMA ---
y0 = 0.1
tempo_inicio = 0
tempo_fim = 10

# 1. SOLUÇÃO ANALÍTICA (Para comparação)
def solucao_analitica(t):
    return 1 / (1 + 9 * np.exp(-t))

# --- ITEM (B): GERANDO DADOS "EXPERIMENTAIS" ---
N = 50
t_i = np.linspace(tempo_inicio, tempo_fim, N) # Aqui definimos o t_i!
ruido = np.random.normal(0, 0.01, N)
y_i = solucao_analitica(t_i) + ruido

# --- ITEM (C): MÉTODO DE EULER ---
h = 0.2 #passo
tempo_euler = np.arange(tempo_inicio, tempo_fim + h, h)
y_euler = np.zeros(len(tempo_euler))
y_euler[0] = y0

# EDO: dy/dt = y(1-y)
for i in range(len(tempo_euler) - 1):
    f_y = y_euler[i] * (1 - y_euler[i])
    y_euler[i+1] = y_euler[i] + h * f_y

# --- PLOTAGEM ---
plt.figure(figsize=(10, 5))

# Dados do item (b)
plt.scatter(t_i, y_i, color='red', alpha=0.4, s=15, label='Dados Experimentais (Item b)')

# Euler do item (c)
plt.plot(tempo_euler, y_euler, 'b-o', markersize=4, label=f'Método de Euler (h={h})')

# Curva teórica perfeita
t_curva = np.linspace(tempo_inicio, tempo_fim, 200)
plt.plot(t_curva, solucao_analitica(t_curva), 'g--', label='Solução Analítica (Teórica)') 

plt.xlabel('Tempo (t)')
plt.ylabel('y(t)')
plt.title('Comparação: Euler vs Dados Ruidosos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()