import numpy as np
import matplotlib.pyplot as plt

import empathicTD as empTD
import offTD as offTD
from policies import Policy, LeftRightPolicy
from models import Model
        
from utils import mom
        
#
# Modele with two states (1, 2)
#
pi = LeftRightPolicy(n = 2, p_right = 1)

mu = LeftRightPolicy(n = 2) # Uniform is default
    
model = Model(features = [1, 2], R = [0,0],
              pi = pi, mu = mu,
              I = [1, 0], discounts = [0.9, 0.9],
              lambdas = [0, 0],
              theta0 = 1,
              alpha = 0.001,
              S0 = 0)  

#
# Modele with fives states (1, 2)
#
pi = LeftRightPolicy(n = 5, p_right = 1)

mu = LeftRightPolicy(n = 5, p_left = 2/3)

model = Model(features = [[1, 0, 0],
                          [1, 1, 0],
                          [0, 1, 0],
                          [0, 1, 1],
                          [0, 0, 1]],
              R = np.ones(5),
              pi = pi, mu = mu,
              v_pi = [4, 3, 2, 1, 1],
              I = np.ones(5),
              discounts = [0, 1, 1, 1, 0],
              lambdas = np.zeros(5),
              theta0 = 0, 
              alpha = 0.001, S0 = 0)  

T = 1000
N = 100
theta_emp, S, F, A = empTD.run(model, T, N)  
theta_neu, S = offTD.run(model, T, N)    

theta_neu_opt = offTD.optimal_run(model, T)
theta_emp_opt = empTD.optimal_run(model, T) 

theta_emp_mom = mom(np.swapaxes(theta_emp, 0, 1))
theta_neu_mom = mom(np.swapaxes(theta_neu, 0, 1))








#
# Plot particle thetas
#

i = 1
plt.title("EmpathicTD and offTD with {} particles".format(N))
plt.plot(theta_emp[:, : , i].squeeze(), linewidth = 0.2, c = "blue")
plt.plot(theta_emp_mom[:, i], linewidth = 3, c = "black", linestyle = "dotted")
plt.plot(theta_emp_opt[:, i], c = "black", linewidth = 3)

plt.plot(theta_neu[:, : , i].squeeze(), linewidth = 0.2, c = "red")
plt.plot(theta_neu_opt[:, i], c = "black", linewidth = 3)
plt.plot(theta_neu_mom[:, i], linewidth = 3, c = "black", linestyle = "dotted")

#plt.ylim(0, 3)
plt.xlim(0, T)
plt.show()



plt.title("MSVE with {} particles".format(N))

plt.plot(model.parallel_msve(theta_emp), linewidth = 0.2, c = "blue")
plt.plot(model.parallel_msve(theta_neu), linewidth = 0.2, c = "red")

plt.plot(model.msve(theta_emp_opt), linewidth = 6, c = "black", linestyle = "dotted")
plt.plot(model.msve(theta_neu_opt), linewidth = 6, c = "black", linestyle = "dotted")

plt.show()



































    



