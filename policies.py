import numpy as np

class Policy(object):
    '''
        Store a markov chain policy
        It basically stores the matrix of transition P.
        It adds helpful function to compute the next steps from multiple states (efficient)
    '''
    
    def __init__(self, P):
        self.P = np.array(P)
        
        self._load_stationary()


    def _load_stationary(self):
        ''' Return the stationary distribution of the markov chain P '''
        vals, vecs = np.linalg.eig(self.P.transpose())
        I = np.where(np.round(vals, 2) == 1)
        i = I[0]
        pi_t = vecs[:, i]
        pi_t = pi_t/pi_t.sum()
        self.d = pi_t.flatten()
        self.D = np.diag(self.d)
        
    def next_step(self, s):
        '''  Pick a state according to the mass distribution of s '''
        P = self.P[s]
        m = len(P)
        return int(np.random.choice(np.arange(m), 1, p = list(P)))
    
    def parallel_steps(self, S):
        ''' Pick the next step for all the states in S (in parallel) '''
        S = S.copy()
        for s in np.unique(S):
            idxs = np.where(S == s)
            n = len(idxs[0]) # Number of chain in state s
            S[idxs] = np.random.choice(2, n, p = self.P[s])
        return S