import numpy as np

# Normalize the data.
from sklearn import preprocessing
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
import scipy.spatial.distance as spdist

def Gp_phy():

    # Physics data
    data_phyloss = np.loadtxt('../data/unlabeled_data_BK_constw_v2_1525.dat')
    x_unlabeled = data_phyloss[:, :]

    x_unlabeled1 = x_unlabeled[:1303, :2]
    x_unlabeled2 = x_unlabeled[-6:, :2]
    y_unlabeled1 = data_phyloss[:1303, -2:-1]
    y_unlabeled2 = data_phyloss[-6:, -2:-1]

    x_unlabeled = np.vstack((x_unlabeled1,x_unlabeled2))
    y_unlabeled = np.vstack((y_unlabeled1,y_unlabeled2))

    # normalize dataset with MinMaxScaler
    scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
    x_unlabeled = scaler.fit_transform(x_unlabeled)

    def covSEard(hyp=None, x=None, z=None):
        ''' Squared Exponential covariance function with Automatic Relevance Detemination
         (ARD) distance measure. The covariance function is parameterized as:

         k(x^p,x^q) = sf2 * exp(-(x^p - x^q)' * inv(P) * (x^p - x^q)/2)

         where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
         D is the dimension of the input space and sf2 is the signal variance.

         The hyperparameters are:

         hyp = [ log(ell_1)
                 log(ell_2)
                 ...
                 log(ell_D)
                 log(sqrt(sf2)) ]
        '''

        [n, D] = x.shape
        ell = 1/np.array(hyp[0:D])        # characteristic length scale
        sf2 = np.array(hyp[D])**2         # signal variance
        tmp = np.dot(np.diag(ell),x.T).T
        A = spdist.cdist(np.dot(np.diag(ell),x.T).T, np.dot(np.diag(ell),z.T).T, 'sqeuclidean') # cross covariances
        A = sf2*np.exp(-0.5*A)  

        return A


    def posterior_predictive(X_s, X_train, Y_train, l1=.1, l2=.1, sigma_f=.1, sigma_y=0):
        '''  
        Computes the suffifient statistics of the GP posterior predictive distribution 
        from m training data X_train and Y_train and n new inputs X_s.

        Args:
            X_s: New input locations (n x d).
            X_train: Training locations (m x d).
            Y_train: Training targets (m x 1).
            l: Kernel length parameter.
            sigma_f: Kernel vertical variation parameter.
            sigma_y: Noise parameter.

        Returns:
            Posterior mean vector (n x d) and covariance matrix (n x n).
        '''
        K = covSEard(hyp=[l1,l2,sigma_f], x=X_train, z=X_train) + sigma_y**2 * np.eye(len(X_train))
        K_s = covSEard(hyp=[l1,l2,sigma_f], x=X_train, z=X_s)
        K_ss = covSEard(hyp=[l1,l2,sigma_f], x=X_s, z=X_s)  + 1e-8 * np.eye(len(X_s))
#         K_inv = inv(K)
        K_inv = np.linalg.pinv(K)
    
        # Equation (4)
        mu_s = K_s.T.dot(K_inv).dot(Y_train)

        # Equation (5)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        
        return mu_s, cov_s


    def nll_fn(X_train, Y_train, noise=0, naive=False):
        '''
        Returns a function that computes the negative log marginal
        likelihood for training data X_train and Y_train and given 
        noise level.

        Args:
            X_train: training locations (m x d).
            Y_train: training targets (m x 1).
            noise: known noise level of Y_train.
            naive: if True use a naive implementation of Eq. (7), if 
                   False use a numerically more stable implementation. 

        Returns:
            Minimization objective.
        '''

        def nll_stable(theta):
            # Numerically more stable implementation of Eq. (7) as described
            # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
            # 2.2, Algorithm 2.1.
            K = covSEard(hyp=[theta[0],theta[1],theta[2]], x=X_train, z=X_train) + \
                noise**2 * np.eye(len(X_train))
            
            K += 1e-6 * np.eye(*K.shape)
            L = cholesky(K)
            return np.sum(np.log(np.diagonal(L))) + \
                   0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) + \
                   0.5 * len(X_train) * np.log(2*np.pi)

        if naive:
            return nll_naive
        else:
            return nll_stable
    
    # Optimization - GP1 - for physics
    res1 = minimize(nll_fn(x_unlabeled, y_unlabeled), x0 = [.1, .1, .1], 
                   bounds=((1e-5, None), (1e-5, None), (1e-5, None)),
                    method='L-BFGS-B')
    
    # GP (physics) predictions for the physics sets
    mu_phy, cov_phy = posterior_predictive(x_unlabeled, x_unlabeled, y_unlabeled, *res1.x)
    
    return [res1, mu_phy]