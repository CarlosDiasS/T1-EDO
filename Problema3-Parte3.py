import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURAÇÕES E PARÂMETROS ---
N = 50                          # Número de pontos (item b)
tempo_inicio = 0
tempo_fim = 10
y0 = 0.1                        # Condição inicial
# Cálculo do passo h baseado na grade de tempo
t_i = np.linspace(tempo_inicio, tempo_fim, N)
h = t_i[1] - t_i[0] 

# Solução analítica real: y(t) = 1 / (1 + 9*exp(-t))
def solucao_analitica(t):
    return 1 / (1 + 9 * np.exp(-t))

# --- 2. GERAÇÃO DOS DADOS "EXPERIMENTAIS" (Item b) ---
np.random.seed(42) # Para você ter sempre os mesmos resultados ao testar
ruido = np.random.normal(0, 0.01, N)
y_i = solucao_analitica(t_i) + ruido

# --- 3. RESOLUÇÃO DOS MÍNIMOS QUADRADOS (Item e) ---
# Matriz A: [h, h*y_i, h*y_i^2]
A = np.column_stack([
    np.full(len(y_i)-1, h), 
    h * y_i[:-1], 
    h * (y_i[:-1]**2)
])

# Vetor r: y_{i+1} - y_i
r = np.diff(y_i)

# Resolvendo: A * a = r
a_hat, _, _, _ = np.linalg.lstsq(A, r, rcond=None)

# --- 4. VALIDAÇÃO: MÉTODO DE EULER COM O MODELO APRENDIDO (Item f) ---
y_simulado = np.zeros(len(t_i))
y_simulado[0] = y0

for i in range(len(t_i) - 1):
    # f_a(y) = a0 + a1*y + a2*y^2
    derivada_aprendida = a_hat[0] + a_hat[1]*y_simulado[i] + a_hat[2]*(y_simulado[i]**2)
    y_simulado[i+1] = y_simulado[i] + h * derivada_aprendida

# --- 5. VISUALIZAÇÃO DOS RESULTADOS ---
plt.figure(figsize=(12, 6))

# Dados coletados (com ruído)
plt.scatter(t_i, y_i, color='red', alpha=0.5, s=20, label='Dados Ruidosos (Entrada)')

# Solução Analítica (O ideal teórico)
t_suave = np.linspace(tempo_inicio, tempo_fim, 200)
plt.plot(t_suave, solucao_analitica(t_suave), 'g-', label='Analítica Real', linewidth=2)

# Modelo Aprendido (O que o Python descobriu)
plt.plot(t_i, y_simulado, 'b--', label='Modelo Aprendido (Euler + MQ)', linewidth=2)

plt.title('Identificação de Sistemas: Recuperando a EDO Logística')
plt.xlabel('Tempo (t)')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True, alpha=0.3)

# Print dos coeficientes para o seu relatório
texto_coeficientes = (f"Coeficientes Aprendidos:\n"
                      f"$a_0 = {a_hat[0]:.4f}$\n"
                      f"$a_1 = {a_hat[1]:.4f}$\n"
                      f"$a_2 = {a_hat[2]:.4f}$")

# Inserindo o texto no gráfico (ajuste as coordenadas 6.0 e 0.2 se necessário)
plt.text(6.0, 0.2, texto_coeficientes, fontsize=11, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))

plt.show()