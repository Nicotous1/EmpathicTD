#
# Utilities functions
#

import numpy as np
import matplotlib.pyplot as plt

def custom_mult(X, m):
    '''
        Custom multiplication for the features of the N particles
        Surely there is another way.
    '''
    return (X.T * m).T



def to_array_of_vectors(X):
    '''
        Assure that the numpy array of features has the good shape.
        Convert (x,) shape to (x,1) if necessary
    '''
    X = np.array(X)
    shape = X.shape
    if len(shape) == 1:
        return X.reshape((shape[0], 1))
    else:
        return X
    
def mom(X, K = 10):
    '''
     Compute the mediam of means with K classes
    '''
    C = np.random.randint(0, K, len(X))
    means = []
    for k in range(K):
        m = np.mean(X[np.where(C == k)], axis = 0)
        means.append(m)
    return np.median(means, axis = 0)




# Empathic library imports
import empathicTD as empTD
import offTD as offTD

class comparatorTD(object):
    '''
        This class below is used to make the notebook clearer.
        It computes the offTD and the empathicTD for a given models.
        It is written in this notebok to explain how you can use the empathicTD library.
        It is also very helpfull for plotting because it uses dynamic limit for y.
        Thus, it deals with the outliers of the empathicTD.
    '''
    
    def __init__(self, model):
        self.model = model
        
        
        
    def run(self, T, N, verbose = True):   
        '''
            Run the offTD and the empTD on the model
            Compute also the deterministic descent
                         the MOM estimator (to remove outlier)
                         the theta optimal
        '''
        
        self.theta_emp = empTD.run(self.model, T, N, verbose = verbose)  
        self.theta_off = offTD.run(self.model, T, N, verbose = verbose)    

        self.theta_off_opt = offTD.optimal_run(self.model, T)
        self.theta_emp_opt = empTD.optimal_run(self.model, T) 

        self.theta_emp_mom = mom(np.swapaxes(self.theta_emp, 0, 1))
        self.theta_off_mom = mom(np.swapaxes(self.theta_off, 0, 1))     
        
        self.theta_final = empTD.optimal(self.model)   
        
        
        
    def plot_theta(self, i = 0, mom = True, particles = True, optimal = True, figure = True, ylim = None):
        '''
            Plot one dimension (i) of theta across the particles
        '''
        T, N, p = self.theta_emp.shape
        
        if figure: plt.figure()

        if particles:
            plt.plot(self.theta_emp[:, : , i].squeeze(), linewidth = 0.2, c = "blue")
            plt.plot(self.theta_off[:, : , i].squeeze(), linewidth = 0.2, c = "red")
        
        if optimal:
            plt.plot(self.theta_emp_opt[:, i], c = "black", linewidth = 3)
            plt.plot(self.theta_off_opt[:, i], c = "black", linewidth = 3)
            
        if mom:
            plt.plot(self.theta_emp_mom[:, i], linewidth = 3, c = "black", linestyle = "dotted")
            plt.plot(self.theta_off_mom[:, i], linewidth = 3, c = "black", linestyle = "dotted")
        
        # Set dynamic limit (to deal with outliers)
        ymin, ymax = self._get_born([self.theta_emp_opt[:, i], self.theta_off_opt[:, i]]) if ylim is None else ylim
        plt.ylim(ymin, ymax)
  
        plt.title("EmpathicTD and offTD with {} particles".format(N))
        plt.xlim(0, T)
        plt.ylabel("theta")
        plt.xlabel("steps")
        
        
        
    def plot_msve(self, figure = True, ylim = None, particles = True, optimal = True):
        '''
            Plot the msve of theta across the particles
        '''
        T, N, p = self.theta_emp.shape
        
        # Compute MSVE
        msve_emp = self.model.parallel_msve(self.theta_emp)
        msve_off = self.model.parallel_msve(self.theta_off)
        msve_emp_opt = self.model.msve(self.theta_emp_opt)
        msve_off_opt = self.model.msve(self.theta_off_opt)
        
        if figure: plt.figure()

        if particles:
            plt.plot(msve_emp, linewidth = 0.2, c = "blue")
            plt.plot(msve_off, linewidth = 0.2, c = "red")
        
        if optimal:
            plt.plot(msve_emp_opt, c = "black", linewidth = 3)
            plt.plot(msve_off_opt, c = "black", linewidth = 3)
        
        # Set dynamic limit (to deal with outliers)
        ymin, ymax = self._get_born([msve_emp_opt, msve_off_opt]) if ylim is None else ylim
        plt.ylim(ymin, ymax)

        plt.title("MSVE with {} particles".format(N))
        plt.ylabel("MSVE")
        plt.xlabel("steps")
        plt.xlim(0, T)
        
        
            
    def _get_born(self, X, margin = 0.1):
        '''
            Return the born for all numpy arrays in X.
            Usefull to find good limit for y in plot.
        '''
        ymin, ymax = [], []
        for x in X:
            ymin.append(x.min())
            ymax.append(x.max())
        ymin, ymax = min(ymin), max(ymax)
        
        l = ymax - ymin
        ymin -= l*margin
        ymax += l*margin
        
        return ymin, ymax