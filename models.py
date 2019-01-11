import numpy as np
from utils import to_array_of_vectors
from scipy.optimize import minimize

class Model(object):
    '''
         A class to store all the parameter of the model
         Features, policies (off ond on)
         Lambdas and discounts for the emphatic TD
    '''
    
    def __init__(self, features, R, pi, theta0, S0, mu = None, I = None, discounts = None, v_pi = None):
        '''
           Set the parameters and compute other parameters to help  
        '''
        self.features = to_array_of_vectors(features) # Features for the state (the function assures it has good shape)
        self.R = np.array(R) # The immediate reward for each state
        self.S0 = int(S0)
        self.theta0 = np.array(theta0)
        
        # Compute basic parameters
        self.n = len(features) # Number of states
        self.p = self.features.shape[1] # Dimension of the features
        
        # Compute policies
        self.pi = pi.fit(self) # The target policy
        self.mu = self.pi if mu is None else mu.fit(self) # The behavior policy (default is the target policy)
        self.phi = np.divide(self.pi.P, self.mu.P, out=np.zeros_like(self.pi.P), where=self.mu.P!=0) # Importance sampling ratio
        
        # Other parameters with default
        self.I = np.ones(self.n)/self.n if I is None else np.array(I) # Intereset for each state (default is uniform)
        self.discounts = np.zeros(self.n) if discounts is None else np.array(discounts) # Discount rate for each state (default is zero for all)
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





class Grid(Model):
    def __init__(self, l_x, l_y, pi, theta0, S0, features = None, R = None, mu = None, I = None, discounts = None, v_pi = None):
        # Grid properties
        self.l_x = int(l_x)
        self.l_y = int(l_y)
        self.n = self.l_x*self.l_y
        
        # Convert for standard model (not 2D but 1D)
        S0 = self.coords_to_id(S0)
        features = np.identity(self.n) if features is None else features
        R = None if R is None else np.tile(R.reshape(self.n), self.n).reshape((self.n, self.n))
        I = None if I is None else np.array(I).flatten()
        discounts = None if discounts is None else np.array(discounts).flatten()
        v_pi = None if v_pi is None else np.array(v_pi).flatten()
        
        
        super(Grid, self).__init__(features, R, pi, theta0, S0,  mu = mu, I = I, discounts = discounts, v_pi = v_pi)
        
        
    def coords_to_id(self, pos):
        if pos is None:
            return False
        x, y = pos
        if (0 > x) or (x >= self.l_x) or (0 > y) or (y >= self.l_y):
            return False
        x = x % self.l_x
        y = y % self.l_y
        return x * self.l_y + y