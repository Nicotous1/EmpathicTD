import numpy as np

from utils import custom_mult

class AbstractTD(object):
    '''
        Abstract class for all TD algorithm.
        It should be inherited by the algorithm class
    '''
    
    def __init__(self, alpha, lambdas):
        self.alpha = alpha
        self.lambdas = lambdas


    def optimal(self, model):
        '''
            Return the optimal theta for the model
        '''
        A, b = self.key_matrixes(model)
        return np.dot(np.linalg.inv(A), b)
    
    
    
    def optimal_run(self, model, T):
        '''
            Return the optimal descent with the key matrix of the model
        '''
        A, b = self.key_matrixes(model)
        thetas = np.zeros((T+1, model.p))
        thetas[0] = model.theta0
        for t in range(0, T):
            thetas[t+1] = thetas[t] + self.alpha * (b - np.dot(A, thetas[t]))
        return thetas
     
    def _get_lambda(self, model):
        '''
         Return a matrix of lambdas of good shape for the model (a lambda for each state)
        '''
        try:
            # Lambdas is a value for each state
            len(self.lambdas) # if raise error this is not a list or an array
            return np.array(self.lambdas)
        except:
            # Lambdas is a number
            n = model.n
            return np.array([self.lambdas]*n)
            
     
    

class OffTD(AbstractTD):
    
    def run(self, model, T, N = 1, verbose = True):
        '''
         Compute the emphatic TD with T period for the model.
         It can do it for N particles in parallel.
        '''
        # Shortcut
        m = model
        
        
        # Init memory
        theta = np.zeros((T+1, N, m.p))
        S = np.zeros((T+1, N), dtype = np.int)
        
        # Set t=0
        S[0] = m.S0
        theta[0] = m.theta0
        
        # Iterating over t (in parallel for the N particles)
        for t in range(T):
            if verbose and (t % 999 == 0):
                print("Computing offTD... ({}/{})".format(t+1, T), end = "\r")
            S[t+1] = m.mu.parallel_steps(S[t]) # Pick next step
                
            # Iterate theta (equation 1)
            # delta is the parathesis of equation 1
            delta = m.R[S[t], S[t+1]]\
                    + m.discounts[S[t+1]] * np.sum(theta[t] * m.features[S[t+1]], axis = 1)\
                    - np.sum(theta[t] * m.features[S[t]], axis = 1)
            theta[t+1] = theta[t] + custom_mult(m.features[S[t]], self.alpha * m.phi[S[t], S[t+1]] * delta)
        
        if verbose:
            print("offTD has been computed for {} steps and {} particles.".format(T, N))    
        
        return theta
    
    
    def key_matrixes(self, model):
        '''
            Compute the matrix A and b for the model
        '''
        Id = np.eye(model.n)
        
        gammas = np.diag(model.discounts)
        
        # Computing A
        # First P_pi_lambda
        A = Id - np.dot(model.pi.P, gammas)
        A = np.dot(model.mu.D, A)
        A = np.dot(np.dot(model.features.transpose(), A), model.features)
        
        # Computing B
        r_pi = np.sum(model.pi.P * model.R, axis = 1)
        B = np.dot(np.dot(model.mu.D, model.features).transpose(), r_pi)    
            
        return A, B    
    
    
    
class EmphaticTD(AbstractTD):
    
    def run(self, model, T, N = 1, verbose = True):
        '''
         Compute the emphatic TD with T period for the model.
         It can do it for N particles in parallel.
        '''
        # Shortcut
        m = model
        lambdas = self._get_lambda(model)
        
        
        # Init memory
        theta = np.zeros((T+1, N, m.p))
        F = np.zeros((T+1, N))
        M = np.zeros(N)
        E = np.zeros((T+1, N, m.p))
        S = np.zeros((T+1, N), dtype = np.int)
        
        # Set t=0
        S[0] = m.S0
        F[0] = m.I[S[0]]
        theta[0] = m.theta0
        
        # Iterating over t (in parallel for the N particles)
        for t in range(T):
            if verbose and (t % 999 == 0):
                print("Computing emphatic TD... ({}/{})".format(t+1, T), end = "\r")
            S[t+1] = m.mu.parallel_steps(S[t]) # Pick next step
            
             # Compute F (equation 20)
            if t > 0:
                F[t] = m.phi[S[t-1], S[t]] * m.discounts[S[t]] * F[t-1] + m.I[S[t]]
            
            # Compute M (equation 19)
            M = lambdas[S[t]] * m.I[S[t]] + (1 - lambdas[S[t]]) * F[t]
            
            # Compute E (equation 18)
            # Use custom_mult to multiply accross the particle
            E[t] = custom_mult(m.features[S[t]], m.phi[S[t], S[t+1]] * M) # (E for t = 0)
            if t > 0:
                E[t] += custom_mult(E[t-1], m.phi[S[t], S[t+1]] * m.discounts[S[t]] * lambdas[S[t]])
                
            # Iterate theta (equation 17)
            # delta is the parathesis of equation 17
            delta = m.R[S[t], S[t+1]]\
                    + m.discounts[S[t+1]] * np.sum(theta[t] * m.features[S[t+1]], axis = 1)\
                    - np.sum(theta[t] * m.features[S[t]], axis = 1)
        
            theta[t+1] = theta[t] + custom_mult(E[t], self.alpha * delta)
        
        if verbose:
            print("emphatic TD has been computed for {} steps and {} particles.".format(T, N))  
            
        return theta
            
    
    def key_matrixes(self, model):
        '''
            Compute the matrix A and b for the model
        '''
        Id = np.eye(model.n)
        
        i = model.mu.d*model.I
        gammas = np.diag(model.discounts)
        lambdas = np.diag(self._get_lambda(model))
        
        # Computing A
        # First P_pi_lambda
        a = Id - np.dot(model.pi.P, gammas)
        b = Id - np.dot(np.dot(model.pi.P, gammas), lambdas)
        b = np.linalg.inv(b)
        P_pi_lambda = Id - np.dot(b, a)
        
        # Then M
        m = np.dot(np.linalg.inv(Id - P_pi_lambda.transpose()), i)
        M = np.diag(m)
        
        # Product to have A
        A = np.dot(np.dot(M, b), a)
        A = np.dot(np.dot(model.features.transpose(), A), model.features)
        
        # Computing B
        B = np.dot(model.features.transpose(), np.dot(M, b))
        r_pi = np.sum(model.pi.P * model.R, axis = 1)
        B = np.dot(B, r_pi)
            
        return A, B