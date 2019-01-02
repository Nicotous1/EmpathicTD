import numpy as np

from utils import custom_mult

def run(model, T, N = 1, alpha = 0.001, S0 = 0, theta0 = 1):
    '''
     Compute the empathic TD with T period for the model.
     It can do it for N particles in parallel.
    '''
    # Shortcut
    m = model
    
    
    # Init memory
    theta = np.zeros((T+1, N, m.p))
    F = np.zeros((T+1, N))
    M = np.zeros(N)
    E = np.zeros((T+1, N, m.p))
    S = np.zeros((T+1, N), dtype = np.int)
    
    # Set t=0
    S[0] = S0
    F[0] = m.I[S[0]]
    theta[0] = theta0
    
    # Iterating over t (in parallel for the N particles)
    for t in range(T):
        S[t+1] = m.mu.parallel_steps(S[t]) # Pick next step
        
         # Compute F (equation 20)
        if t > 0:
            F[t] = m.phi[S[t-1], S[t]] * m.discounts[S[t]] * F[t-1] + m.I[S[t]]
        
        # Compute M (equation 19)
        M = m.lambdas[S[t]] * m.I[S[t]] + (1 - m.lambdas[S[t]]) * F[t]
        
        # Compute E (equation 18)
        # Use custom_mult to multiply accross the particle
        E[t] = custom_mult(m.features[S[t]], m.phi[S[t], S[t+1]] * M) # (E for t = 0)
        if t > 0:
            E[t] += custom_mult(E[t-1], m.phi[S[t], S[t+1]] * m.discounts[S[t]] * m.lambdas[S[t]])
            
        # Iterate theta (equation 17)
        # delta is the parathesis of equation 17
        delta = m.R[S[t+1]]\
                + m.discounts[S[t+1]] * np.sum(theta[t] * m.features[S[t+1]], axis = 1)\
                - np.sum(theta[t] * m.features[S[t]], axis = 1)
    
        theta[t+1] = theta[t] + custom_mult(E[t], alpha * delta)
        
    return theta, F, E, S    
        

def key_matrixes(model):
    '''
        Compute the matrix A and b for the model
    '''
    Id = np.eye(model.N_states)
    
    i = model.mu.d*model.I
    gammas = np.diag(model.discounts)
    lambdas = np.diag(model.lambdas)
    
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
    B = np.dot(B, model.R)
        
    return A, B



def optimal(model):
    '''
        Return the optimal theta for the model
    '''
    A, b = key_matrixes(model)
    return np.dot(np.linalg.inv(A), b)



def optimal_run(model, T, alpha = 0.001, S0 = 0, theta0 = 1):
    '''
        Return the optimal descent with the key matrix of the model
    '''
    A, b = key_matrixes(model)
    thetas = np.zeros(T+1)
    thetas[0] = theta0
    for t in range(0, T):
        thetas[t+1] = thetas[t] + alpha * (b - np.dot(A, thetas[t]))
    return thetas