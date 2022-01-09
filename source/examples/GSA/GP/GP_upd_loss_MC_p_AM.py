import scipy.io as spio
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
def pass_arg_upd(Xx, nsim, tr_size, pre_trained_hyperparamters):

    print("tr_Size:",tr_size)
    tr_size = int(tr_size)
    # Making sure final porosity is less than initial
    def poros(poroi, porof):
        porofn = -porof*(porof<0)
        porofp = porof*(porof>=poroi) - poroi*(porof>=poroi)
        return porofp+porofn

    
    # Load labeled data
    data = np.loadtxt('../../../../data/labeled_data.dat')
    x_labeled = data[:, :2].astype(np.float64) # -2 because we do not need porosity predictions
    y_labeled = data[:, -2:-1].astype(np.float64) # dimensionless bond length and porosity measurements

    # normalize dataset with MinMaxScaler
    scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
    x_labeled = scaler.fit_transform(x_labeled)
    scaler_y = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
    y_labeled = scaler_y.fit_transform(y_labeled)

    tr_size = int(tr_size)


    data_phyloss = np.loadtxt('../../../../data/unlabeled_data_BK_constw_v2_1525.dat')
    x_unlabeled = data_phyloss[:, :]

    # initial porosity
    initporo = x_unlabeled[:, -1]

    x_unlabeled1 = x_unlabeled[:1303, :2]
    x_unlabeled2 = x_unlabeled[-6:, :2]
    x_unlabeled = np.vstack((x_unlabeled1,x_unlabeled2))

    x_unlabeled = scaler.fit_transform(x_unlabeled)
    init_poro1 = initporo[:1303]
    init_poro2 = initporo[-6:]
    init_poro = np.hstack((init_poro1,init_poro2))
    
    
    # train and test data
    trainX, testX, trainY, testY = train_test_split(x_labeled, y_labeled, train_size=tr_size/x_labeled.shape[0], 
                                                    random_state=42, shuffle=True)

           
        
    def log_marginal_likeli(self, theta=None, eval_gradient=False,
                                clone_kernel=True):
        """Returns log-marginal likelihood of theta for training data.
        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,) default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L, True), y_train)  # Line 3

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
        
        
        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        
        # regularization term added
        pred1 = self.predict(x_unlabeled)
        phyloss = np.mean(poros(init_poro, pred1))
        pred1 = scaler_y.inverse_transform(pred1)
        #print("phyLoss:", 500*phyloss)
        log_likelihood -= 500000*phyloss
        #print(log_likelihood)
        
        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,jik->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood
        
        
        
    def obj_funct(theta, eval_gradient=True):
        if eval_gradient:
            lml, grad = log_marginal_likeli(gp2, theta, eval_gradient=True, clone_kernel=False)
            return -lml, -grad
        else:
            return -log_marginal_likeli(gp2, theta, clone_kernel=False)
    
    def custom_optimizer(obj_func, initial_theta, bounds):
        custom_optimizer_method = fmin_l_bfgs_b(obj_funct, x0=initial_theta, bounds=bounds)
        # custom_optimizer_method[0]: optimized values can be accessed using
        # custom_optimizer_method[1]: resulting value of the function to be minimized
        return (custom_optimizer_method[0], custom_optimizer_method[1])
        
        
        
    gp2 = GaussianProcessRegressor(kernel=pre_trained_hyperparamters, alpha =.1, n_restarts_optimizer=10)
    gp2.fit(trainX, trainY)

    samples = gp2.sample_y(Xx, n_samples=int(nsim)).T
    return np.squeeze(samples)