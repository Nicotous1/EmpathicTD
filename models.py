import numpy as np
from utils import to_array_of_vectors

class Model(object):
    '''
         A class to store all the parameter of the model
         Features, policies (off ond on)
         Lambdas and discounts for the empathic TD
    '''
    
    def __init__(self, features, R, pi, mu = None, I = None, lambdas = None, discounts = None):
        '''
           Set the parameters and compute other parameters to help  
        '''
        self.features = to_array_of_vectors(features) # Features for the state (the function assures it has good shape)
        self.R = np.array(R) # The immediate reward for each state
        self.pi = pi # The target policy
        self.mu = self.pi if mu is None else mu # The behavior policy (default is the target policy)
        
        # Compute parameters
        self.phi = self.pi.P / self.mu.P # Importance sampling ratio
        self.N_states = len(features) # Number of states
        self.p = self.features.shape[1] # Dimension of the features
        
        self.I = np.ones(self.N_states)/self.N_states if I is None else np.array(I) # Intereset for each state (default is uniform)
        self.lambdas = np.zeros(self.N_states) if lambdas is None else np.array(lambdas) # bootsrap ratio for each state (default is zeros for all)
        self.discounts = np.zeros(self.N_states) if discounts is None else np.array(discounts) # Discount rate for each state (default is zero for all)