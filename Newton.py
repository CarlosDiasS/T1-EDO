import numpy as np
import matplotlib.pyplot as plt

def Resfriamento_euler(T0, Tamb, k, dt, tempo_total):
    passos= int(tempo_total/dt)
    t_valores = np.linspace(0, tempo_total, passos)
    T_valores = np.zeros(passos)
    T_valores[0] = T0

    for i in range(1, passos):
        dTdt = -k * (T_valores[i-1]- Tamb)
        T_valores[i] = T_valores[i-1] + dTdt * dt

    return t_valores, T_valores

k_agua = 0.05
t, T_num = Resfriamento_euler(T0=80, Tamb=25, k=k_agua, dt=0.1, tempo_total=100)

plt.figure(figsize=(10, 6))
plt.plot(t, T_num, label='Simulação (Euler)', color='red')
plt.axhline(y=30, color='blue', linestyle='--', label='Temp. Ambiente (Asseclot)')
plt.title('Resfriamento de Newton - Simulação Numérica')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()