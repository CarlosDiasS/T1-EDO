from math import * 
import pandas as pd





def linearRegression(alfa,beta,x,y):

    n = len(x)
    
    y_prev = 0.0
    mse = 0.0
    
    #media dos valores de y 
    y_med = sum(y)/n
    
    #somatoria dos quadrados dos residuos 
    ss_res = 0.0
    
    #total dos quadrados 
    ss_tot = 0.0
    
    for i in range(x):
        y_prev+= alfa*x[i] + beta
        mse+=((y[i] - y_prev))**2
        ss_res += (y[i] - y_prev)**2
        ss_tot += (y[i] - y_med)**2
        
    mse = mse/n
    r2 = 1 - (ss_res/ss_tot)

    return mse,r2
    
    



