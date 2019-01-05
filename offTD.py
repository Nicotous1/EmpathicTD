import numpy as np

from utils import custom_mult

def run(model, T, N = 1, verbose = True):
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
        theta[t+1] = theta[t] + custom_mult(m.features[S[t]], m.alpha * m.phi[S[t], S[t+1]] * delta)
    
    if verbose:
        print("offTD has been computed for {} steps and {} particles.".format(T, N))    
    
    return theta  
        

def key_matrixes(model):
    '''
        Compute the matrix A and b for the model
    '''
    Id = np.eye(model.N_states)
    
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



def optimal(model):
    '''
        Return the optimal theta for the model
    '''
    A, b = key_matrixes(model)
    return np.dot(np.linalg.inv(A), b)



def optimal_run(model, T):
    '''
        Return the optimal descent with the key matrix of the model
    '''
    A, b = key_matrixes(model)
    thetas = np.zeros((T+1, model.p))
    thetas[0] = model.theta0
    for t in range(0, T):
        thetas[t+1] = thetas[t] + model.alpha * (b - np.dot(A, thetas[t]))
    return thetas