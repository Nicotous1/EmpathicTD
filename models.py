import numpy as np
from utils import to_array_of_vectors
from scipy.optimize import minimize

class Model(object):
    '''
         A class to store all the parameter of the model
         Features, policies (off ond on)
         Lambdas and discounts for the emphatic TD
    '''
    
    def __init__(self, features, R, pi, theta0, S0, alpha = 0.001, mu = None, I = None, lambdas = None, discounts = None, v_pi = None):
        '''
           Set the parameters and compute other parameters to help  
        '''
        self.features = to_array_of_vectors(features) # Features for the state (the function assures it has good shape)
        self.R = np.array(R) # The immediate reward for each state
        self.pi = pi # The target policy
        self.mu = self.pi if mu is None else mu # The behavior policy (default is the target policy)
        self.S0 = int(S0)
        self.theta0 = np.array(theta0)
        
        # Compute parameters
        self.phi = np.divide(self.pi.P, self.mu.P, out=np.zeros_like(self.pi.P), where=self.mu.P!=0) # Importance sampling ratio
        self.N_states = len(features) # Number of states
        self.p = self.features.shape[1] # Dimension of the features
        
        # Other parameters with default
        self.I = np.ones(self.N_states)/self.N_states if I is None else np.array(I) # Intereset for each state (default is uniform)
        self.lambdas = np.zeros(self.N_states) if lambdas is None else np.array(lambdas) # bootsrap ratio for each state (default is zeros for all)
        self.discounts = np.zeros(self.N_states) if discounts is None else np.array(discounts) # Discount rate for each state (default is zero for all)
        self.v_pi = None if v_pi is None else np.array(v_pi)
        
#
# Utilities
#        
        
    def msve_min(self):
        '''
            Compute the minimum of the msve which exist because it is convex.
            Return x_min, min
        '''
        def msve(theta):
            return np.sum((np.dot(self.features, theta) - self.v_pi)**2 * self.mu.d * self.I)
        m = minimize(msve, self.theta0)
        return m.x, m.fun
        
    def msve(self, theta):
        '''
            Compute the MSVE for only one particle
        '''
        if self.v_pi is None:
            raise ValueError("v_pi must be defined to compute the msve !")
        v_estimates = np.dot(theta, self.features.transpose())
        msve = ((v_estimates - self.v_pi)**2) * self.mu.d * self.I
        msve = np.sum(msve, axis = 1)
        return msve
    
    def parallel_msve(self, thetas):
        '''
            Compute the MSVE for multiple particles in parallel
        '''
        if self.v_pi is None:
            raise ValueError("v_pi must be defined to compute the msve !")
        v_estimates = np.tensordot(self.features, thetas, axes = [[1], [2]])
        v_estimates = np.moveaxis(v_estimates, 0, -1) # Change axis [S, T, N] -> [T, N, S]
        msve = ((v_estimates - self.v_pi)**2) * self.mu.d * self.I
        msve = np.sum(msve, axis = 2)
        #msve = np.linalg.norm((v_estimates - self.v_pi), axis = 2)
        return msve
        