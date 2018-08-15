# python3 best_fit_line.py

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')

#xs = np.array([1,2,3,4,5,6] , dtype = np.float64)
#ys = np.array([5,4,6,5,6,7] , dtype = np.float64)

def create_dataset(n, variance, step=2, correlation="pos"):
    val = 1
    ys = []
    for i in range(n):
        y = val + random.randrange(-variance , variance)
        print(val , y)
        ys.append(y)
        if correlation=="pos":
            val += step
        elif correlation=="neg":
            val -= step
    xs = [i for i in range(len(ys))] #xs = [0,1,2,3,4,5,6,.......]
    return np.array(xs , dtype = np.float64) , np.array(ys , dtype = np.float64)


def best_slope_intercept():
    m = ( (mean(xs)*mean(ys) - mean(xs*ys)) /
        (mean(xs)**2 - mean(xs*xs)) )
    
    b = mean(ys) - m*mean(xs)
    return b,m

def squared_error(ys_orig_line , ys_reg_line):
    e = sum((ys_orig_line - ys_reg_line)**2)
    return e

def coef_of_determination(ys_orig_line , ys_reg_line):
    ys_mean_line = [mean(ys) for y in ys]
    e_reg_line = squared_error(ys_orig_line , ys_reg_line)
    e_mean_line = squared_error(ys_orig_line , ys_mean_line)

    coef = 1 - (e_reg_line)/(e_mean_line)
    return coef


xs,ys = create_dataset(10, 5,2,"pos")

b,m = best_slope_intercept()

regression_line = [ (m*x)+b for x in xs ]

r2 = coef_of_determination(ys , regression_line)
print(r2)

xx = 5
yy = m*xx + b

plt.scatter(xx,yy,color="red",s=50)

plt.scatter(xs,ys,color="blue")
plt.plot(xs,regression_line , color = "green")
plt.show()
