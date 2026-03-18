from math import * 
import pandas as pd





def linearRegression(alfa,beta,x,y):

    n = len(x)
    
    y_prev = 0.0
    
    for i in range(x):
        y_prev+= alfa*x[i] + beta


    # metrica MSE 
    mse = 0.0
    for i in range(n):
        mse += sum((y[i] - y_prev))**2 / n

    #metrica r2
    
    #aux1 = 



