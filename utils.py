#
# Utilities functions
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines



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




class comparatorTD(object):
    '''
        This class below is used to make the notebook clearer.
        It computes the offTD and the emphaticTD for a given models.
        It is written in this notebok to explain how you can use the emphaticTD library.
        It is also very helpfull for plotting because it uses dynamic limit for y.
        Thus, it deals with the outliers of the emphaticTD.
    '''
    
    def __init__(self, algos, colors = None, names = None):
        self.algos = algos
        self.colors = ["black"]*len(algos) if colors is None else colors
        self.names = ["Algo {}".format(i+1) for i in range(len(algos))] if names is None else names
        
        self.res = None
        
        
        
        
    def run(self, model, T, N, verbose = True):   
        '''
            Run the offTD and the empTD on the model
            Compute also the deterministic descent
                         the MOM estimator (to remove outlier)
                         the theta optimal
        '''
        self.model = model
        
        self.res = []
        for algo in self.algos:
            theta = algo.run(model, T, N, verbose = verbose)   
            theta_opt = algo.optimal_run(model, T)
            theta_mom = mom(np.swapaxes(theta, 0, 1))
            #theta_final = algo.optimal(self.model)  
            
            self.res.append((theta, theta_opt, theta_mom))
        
        
        
    def plot_theta(self, i = 0, mom = True, particles = True, optimal = True, figure = True, ylim = None):
        '''
            Plot one dimension (i) of theta across the particles
        '''
        
        if figure: plt.figure()
        
        legends = []
        
        ymin, ymax = None, None
        for res, algo, color, name in zip(self.res, self.algos, self.colors, self.names):
            theta, theta_opt, theta_mom = res
            T, N, p = theta.shape
            

            if particles:
                plt.plot(theta[:, : , i].squeeze(), linewidth = 0.2, c = color)
                legends.append(mlines.Line2D([], [], color=color, label=name))
        
            color_others = "black" if particles else color
            if optimal:
                plt.plot(theta_opt[:, i], c = color_others, linewidth = 3)
            
            if mom:
                plt.plot(theta_mom[:, i], linewidth = 3, c = color_others, linestyle = "dotted")
            
            if ylim is None:
                # Auto set up of limit
                a_ymin, a_ymax = theta_opt[:, i].min(), theta_opt[:, i].max()
                if ymin is None:
                    ymin, ymax = a_ymin, a_ymax
                else:
                    ymin, ymax = min(ymin, a_ymin), max(ymax, a_ymax)
        
        
        if optimal:
            legends.append(mlines.Line2D([], [], color='black', label='deterministic', linewidth = 3))
        if mom:
            legends.append(mlines.Line2D([], [], color='black', linestyle="dotted", label='MOM', linewidth = 3))
            
        # Set dynamic limit (to deal with outliers)
        if not(ylim is None):
            ymin, ymax = ylim
        else:
            ymin, ymax = self._add_margin(ymin, ymax)
        plt.ylim(ymin, ymax)
  
    
        plt.title("TD with {} particles".format(N))
        plt.xlim(0, T)
        plt.ylabel("theta")
        plt.xlabel("steps")
        plt.legend(handles=legends)
        
        
        
    def plot_msve(self, figure = True, ylim = None, particles = True, optimal = True, mom = True):
        '''
            Plot the msve of theta across the particles
        '''        
        if figure: plt.figure()
        
        legends = []
        
        ymin, ymax = None, None
        
        for res, algo, color, name in zip(self.res, self.algos, self.colors, self.names):
            theta, theta_opt, theta_mom = res
            T, N, p = theta.shape
            
            # Compute msve
            msve = self.model.parallel_msve(theta)
            msve_opt = self.model.msve(theta_opt)
            msve_mom = self.model.msve(theta_mom)
            
            legends.append(mlines.Line2D([], [], color=color, label=name))
            
            if particles:
                plt.plot(msve, linewidth = 0.2, c = color)
        
            color_others = "black" if particles else color
            if optimal:
                plt.plot(msve_opt, c = color_others, linewidth = 3)
            
            if mom:
                plt.plot(msve_mom, linewidth = 3, c = color_others, linestyle = "dotted")
            
            # Auto set up of limit
            if ylim is None:
                a_ymin, a_ymax = msve_opt.min(), msve_opt.max()
                if ymin is None:
                    ymin, ymax = a_ymin, a_ymax
                else:
                    ymin, ymax = min(ymin, a_ymin), max(ymax, a_ymax)
        
        
        if optimal and (particles or mom):
            legends.append(mlines.Line2D([], [], color='black', label='deterministic', linewidth = 3))
            
        if mom and (particles or optimal):
            legends.append(mlines.Line2D([], [], color='black', label='MOM', linewidth = 3, linestyle = "dotted"))
            
            
        
        # Set dynamic limit (to deal with outliers)
        if not(ylim is None):
            ymin, ymax = ylim
        else:
            ymin, ymax = self._add_margin(ymin, ymax)
        plt.ylim(ymin, ymax)

        plt.title("MSVE with {} particles".format(N))
        plt.ylabel("MSVE")
        plt.xlabel("steps")
        plt.xlim(0, T)
        plt.legend(handles=legends)
        
        
        
    def _add_margin(self, ymin, ymax, margin = 0.1):
        l = ymax - ymin
        margin = l*margin
        return ymin - margin, ymax + margin
        