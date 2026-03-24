import numpy as np
import matplotlib.pyplot as plt

def Resfriamento_euler(T0, Tamb, k, dt, tempo_total):
    passos= int(tempo_total/dt)
    t_valores = np.linspace(0, tempo_total, passos)
    T_valores = np.zeros(passos)
    T_valores[0] = T0

    for i in range(1, passos):
        dTdt = -k * (T_valores[i-1]- Tamb)
        T_valores[i] = (T_valores[i-1] + k*dt*Tamb) / (1 + k*dt)
        
        if T_valores[i] < Tamb:
            T_valores[i] = Tamb

    return t_valores, T_valores

k_agua = 0.05
t, T_num = Resfriamento_euler(T0=60, Tamb=23.5, k=k_agua, dt=0.01, tempo_total=100)

t_exp = np.array([0, 8, 9, 10, 11, 15, 20, 25, 27])
T_exp = np.array([61.5, 54.3, 53.6, 52.8, 52.0, 49.6, 46.9, 44.4, 43.3])


plt.figure(figsize=(10, 6))
plt.plot(t, T_num, label='Simulação (Euler)', color='red')
plt.scatter(t_exp, T_exp, label='Dados experimentais', marker='o')
plt.plot(t_exp, T_exp, linestyle='--') 
plt.axhline(y=23.5, color='blue', linestyle='--', label='Temp. Ambiente')
plt.title('Resfriamento de Newton - Simulação Numérica')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()