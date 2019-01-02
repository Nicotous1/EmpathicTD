import numpy as np
import matplotlib.pyplot as plt

import empathicTD as empTD
import offTD as offTD
from policies import Policy, LeftRightPolicy
from models import Model
        
        

pi = LeftRightPolicy(n = 2, p_right = 1)

mu = LeftRightPolicy(n = 2) # Uniform is default
    
model = Model(features = [[1, 0], [0, 1]], R = [3,10],
              pi = pi, mu = mu,
              I = [0.5,0.5], discounts = [0.9, 0.9],
              lambdas = [0.5,0.5])  

T = 10000
N = 500
theta_emp, S, F, A = empTD.run(model, T, N)  
theta_neu, S = offTD.run(model, T, N)    

theta_neu_opt = offTD.optimal_run(model, T)
theta_emp_opt = empTD.optimal_run(model, T)


i = 1
plt.title("EmpathicTD with {} particles".format(N))
plt.plot(theta_emp[:, : , i].squeeze(), linewidth = 0.2, c = "blue")
plt.plot(theta_neu[:, : , i].squeeze(), linewidth = 0.2, c = "red")
plt.plot(theta_neu_opt[:, i], c = "black", linewidth = 3)
plt.plot(theta_emp_opt[:, i], c = "black", linewidth = 3)
plt.ylim(0, 100)
plt.show()


