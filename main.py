import numpy as np
import matplotlib.pyplot as plt

import empathicTD as empTD
from policies import Policy
from models import Model
        
        

pi = Policy([[0, 1],
               [0, 1]])

mu = Policy([[0.5, 0.5],
               [0.5, 0.5]])
    
model = Model(features = [[1],[3]], R = [0,0],
              pi = pi, mu = mu,
              I = [1,0], discounts = [0.9, 0.9])  

T = 10000
N = 500
theta, F, E, S = empTD.run(model, T, N)   

theta_opt = empTD.optimal_run(model, T)


mean = theta.mean(axis = 1)
std = theta.std(axis = 1)

plt.title("EmpathicTD with {} particles".format(N))
plt.plot(theta.squeeze(), linewidth = 0.2, c = "blue")
plt.plot(theta_opt, c = "black", linewidth = 2)
plt.ylim(-3, 3)
plt.show()















