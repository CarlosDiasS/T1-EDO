import numpy as np
import matplotlib.pyplot as plt

h1 = 1
h2 = 5
h3 = 10


max_iter = 1000
Epsilon = 1e8

k = 1 # vai entrar dps

def df(t,h):
    return -k*np.sqrt(h)

def euler(ti, tf ,dt ,h0, max_iter):
    n = int((tf-ti)/dt)
    
    t = np.linspace(ti,tf,n+1)
    h = np.zeros(n+1)
    
    h[0] = h0
    op = 0
    
    for i in range(n-1):
        
        k = df(t[i],h[i])
        h[i+1] = h[i] + h*k
        
        op+=1
        
        #erro
        
        if h[i+1] < 0:
            h[i+1] = 0
            return t,h
        
        if(op>=max_iter):
            return t,h
            
    return t,h