import cvxopt
from cvxopt import matrix
from cvxopt import solvers
import os

# silences verbose printing
solvers.options['show_progress'] = False

import numpy as np

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

def sum_square(X):
    return sum(np.square(X))

def localize(Data):
    # Take an input Data, place into standard form
    #Solve using cvxopt
    
    #%%  Problem data
    k = Data.nComparisons
    n = Data.nDims
    
    Xi = Data.Xi # 2 x k array
    Xj = Data.Xj
    
    #%%  START OF LOCALIZATION FROM PAIRED COMPARISONS ############
    
    ##### SET UP P MATRIX
    P = np.zeros((n+k,n+k))
    P[0:n,0:n] = np.eye(n,n)
    
    ##### SET UP h MATRIX
    b = sum_square(Xi) - sum_square(Xj)
    b = b.reshape(1,k)
    
    b = b.transpose()
    b = -b
    h = np.vstack((b, np.zeros((k,1)))) #2k x 1 vector
    
    #%% set up G matrix
    G11 = 2 * (Xj - Xi)
    G11 = G11.transpose() # k x n
    
    G21 = np.zeros((k,n)) # k x n
    
    G12 = -1 * np.eye(k,k) # k x k
    
    G22 = -1 * np.eye(k,k)
    
    G1 = np.hstack([G11, G12])
    G2 = np.hstack([G21, G22])
    
    G = np.vstack([G1,G2])
    #%% set up q matrix (n + k x 1)
    q1 = np.zeros((n,1))
    q2 = np.ones((k,1))
    q = np.vstack([q1,q2])
    
    #%% Now convert into cvxopt matrix form
    
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    # used like
    with suppress_stdout_stderr():
        sol=cvxopt.solvers.qp(P, q, G, h)
    return sol

class ComparisonData():
    def __init__(self, nComparisons, nDims):
        self.nComparisons = nComparisons
        self.nDims = nDims
        self.Xi = np.zeros((nDims, nComparisons))
        self.Xj = np.zeros((nDims, nComparisons))


class PointData:
  def __init__(self, latent_point, closer_point, farther_point):
    self.latent_point = latent_point
    self.closer_point = closer_point
    self.farther_point = farther_point
