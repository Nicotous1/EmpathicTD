#
# Utilities functions
#

import numpy as np

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
