import numpy as np
import matplotlib.pyplot as plt

h_vetor = [1,5,10]

k = 1 # vai entrar dps

def df(t,h):
    return -k*np.sqrt(h)

def euler(ti, tf ,dt ,h0):
    n = int((tf-ti)/dt)
    
    t = np.linspace(ti,tf,n+1)
    h = np.zeros(n+1)
    
    h[0] = h0
    op = 0
    
    for i in range(n):
        
        aux = df(t[i],h[i])
        h[i+1] = h[i] + dt*aux
        
        op+=1
        
        #caso a derivada seja negativa
        if h[i+1] < 0:
            h[i+1] = 0
        
    return t,h

def main():
    
    for h0 in h_vetor:
        t,h = euler(0,10, 0.01, h0)
        plt.plot(t, h, label=f'h0={h0}')

    plt.xlabel('tempo')
    plt.ylabel('altura')
    plt.legend()
    plt.grid()
    plt.show()
    
main()