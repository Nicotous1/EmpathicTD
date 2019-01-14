import numpy as np
import itertools

class Policy(object):
    '''
        Store a markov chain policy
        It basically stores the matrix of transition P.
        It adds helpful function to compute the next steps from multiple states (efficient)
    '''
    
    def __init__(self, P):
        '''
            Just store P and compute its stationary distrubution
        '''
        self.P = np.array(P)
        
        self._load_stationary()
        
        return self


    def fit(self, model):
        '''Do nothing here because P is already defined
            Else it should compute P according to the model
            Must return the fitted policy !!'''
        return self

    def _load_stationary(self):
        ''' Return the stationary distribution of the markov chain P '''
#        vals, vecs = np.linalg.eigh(self.P.transpose())
#        I = np.where(np.round(vals, 2) == 1)
#        i = I[0]
#        pi_t = vecs[:, i]
#        pi_t = pi_t/pi_t.sum()
        a = self.P
        for _ in range(15):
            a = np.dot(a, a)
        pi_t = a[0]
        self.d = pi_t.flatten()
        self.D = np.diag(self.d)
        
    def next_step(self, s):
        '''  Pick a state according to the mass distribution of s '''
        P = self.P[s]
        m = len(P)
        return int(np.random.choice(np.arange(m), 1, p = list(P)))
    
    def parallel_steps(self, S):
        ''' Pick the next step for all the states in S (in parallel) '''
        S_new = S.copy()
        for s in np.unique(S):
            idxs = np.where(S == s)
            n = len(idxs[0]) # Number of chain in state s
            S_new[idxs] = np.random.choice(len(self.P), n, p = self.P[s])
        return S_new
    
    def __str__(self):
        if hasattr(self, 'P'): # the policy is defined
            res = "Transition matrix :\n"
            res += str(self.P) + "\n"
            res += "Stationary distribution : {}".format(self.d)
            return res
        else:
            return "This policy has not been fitted yet !"
    
    
    
class LeftRightPolicy(Policy):
    def __init__(self, p_right = None, p_left = None):
        '''Compute the P matrix associate to the right and left policy'''
        
        # Default is uniform
        if p_right is None and p_left is None:
            p_right, p_left = 0.5, 0.5
            
        # Normalize
        if p_right is None:
            p_right = 1 - p_left
        if p_left is None:
            p_left = 1 - p_right
            
        self.p_left, self.p_right = p_left, p_right
            
    def fit(self, model):
        '''Generate the transition matrix
            Return the fitted model'''
        n, p_right, p_left = model.n, self.p_right, self.p_left #shortcut
        # Set P
        P = np.zeros((n,n))
        P[0:-1, 1:] += np.identity(n-1) * p_right
        P[1:, 0:-1] += np.identity(n-1) * p_left
        P[0, 0] = p_left
        P[-1,-1] = p_right        
        return super(LeftRightPolicy, self).__init__(P)
        
        
        
        
class RandomPolicy(Policy):
    def fit(self, model):
        '''
            Generate random markov chain matrix
            Return the policy fitted
        '''
        P = np.random.random((model.n,model.n))
        s = np.sum(P, axis = 1)
        P = (P.T/s).T
        return super(RandomPolicy, self).__init__(P)
        
        
        

class GridRandomWalkPolicy(Policy):

    def __init__(self, p_up = 0, p_down = 0, p_left = 0, p_right = 0):
        self.p_up = p_up
        self.p_down = p_down
        self.p_left = p_left
        self.p_right = p_right
    
    
    def fit(self, model):
        l_x, l_y, n = model.l_x, model.l_y, model.n
        P = np.zeros((n, n))
        for x, y in itertools.product(range(l_x), range(l_y)):
            idx = model.coords_to_id((x, y))
            
            # Up
            id_next = model.coords_to_id((x - 1, y))
            if id_next is False: id_next = idx # If not exists it does not move
            P[idx, id_next] += self.p_up
            
            # Down
            id_next = model.coords_to_id((x + 1, y))
            if id_next is False: id_next = idx # If not exists it does not move
            P[idx, id_next] += self.p_down
            
            # Right
            id_next = model.coords_to_id((x, y + 1))
            if id_next is False: id_next = idx # If not exists it does not move
            P[idx, id_next] += self.p_right
            
            # Left
            id_next = model.coords_to_id((x, y - 1))
            if id_next is False: id_next = idx # If not exists it does not move
            P[idx, id_next] += self.p_left
            
        P = (P.T / np.sum(P, axis = 1)).T
            
        return super(GridRandomWalkPolicy, self).__init__(P)  