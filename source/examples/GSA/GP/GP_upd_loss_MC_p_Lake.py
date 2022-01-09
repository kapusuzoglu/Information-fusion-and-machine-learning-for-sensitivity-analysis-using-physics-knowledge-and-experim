import scipy.io as spio
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
def pass_arg_upd(Xx, nsim, tr_size, pre_trained_hyperparamters):

    print("tr_Size:",tr_size)
    pre_tr_size = 3000
    loss_num = 2000
    tr_size = int(tr_size)

    #List of lakes to choose from
    lake = ['mendota' , 'mille_lacs']
    lake_num = 0  # 0 : mendota , 1 : mille_lacs
    lake_name = lake[lake_num]

    # Load features (Xc) and target values (Y)
    data_dir = '../../../../data/'
    filename = lake_name + '.mat'
    mat = spio.loadmat(data_dir + filename, squeeze_me=True,
    variable_names=['Y','Xc_doy','Modeled_temp'])
    Xc = mat['Xc_doy']
    Y = mat['Y']
    Xc = Xc[:,:-1]
    # train and test data
    trainX, testX, trainY, testY = train_test_split(Xc, Y, train_size=tr_size/Xc.shape[0], 
                                                    test_size=tr_size/Xc.shape[0], random_state=42, shuffle=True)

    ## train and test data
    #trainX, trainY = Xc[:tr_size,:-1], Y[:tr_size]
    #testX, testY = Xc[-50:,:-1], Y[-50:]

    # Loading unsupervised data
    unsup_filename = lake_name + '_sampled.mat'
    unsup_mat = spio.loadmat(data_dir+unsup_filename, squeeze_me=True,
    variable_names=['Xc_doy1','Xc_doy2'])

    uX1 = unsup_mat['Xc_doy1'] # Xc at depth i for every pair of consecutive depth values
    uX2 = unsup_mat['Xc_doy2'] # Xc at depth i + 1 for every pair of consecutive depth values
    uX1 = uX1[range(0,649723,51),:]
    uX2 = uX2[range(0,649723,51),:]
    # uX1 = uX1[:pre_tr_size,:-1]
    # uX2 = uX2[:pre_tr_size,:-1]
    # uY1 = uX1[:pre_tr_size,-1:]
    # uY2 = uX2[:pre_tr_size,-1:]
    
    # for the physics-based loss function
    uX1_L = uX1[pre_tr_size:pre_tr_size+loss_num,:-1]
    uX2_L = uX2[pre_tr_size:pre_tr_size+loss_num,:-1]
    uY1_L = uX1[pre_tr_size:pre_tr_size+loss_num,-1:]
    uY2_L = uX2[pre_tr_size:pre_tr_size+loss_num,-1:]
           
    # kernel = C(5.0, (1e-2, 1e3)) * RBF(length_scale = [1] * trainX.shape[1], length_scale_bounds=(1e-3, 1e4))
    # gp1 = GaussianProcessRegressor(kernel=kernel, alpha =1.2, n_restarts_optimizer=0)
    # gp1.fit(uX1, uY1)
    
    # # pre-trained model parameters
    # pre_trained_hyperparamters = gp1.kernel_
    
   
    #function for computing the density given the temperature(nx1 matrix)
    def density(temp):
        return 1000 * ( 1 - (temp + 288.9414) * (temp - 3.9863)**2 / (508929.2 * (temp + 68.12963) ) )

    def density_diff(densityf, densityi):
        diff = densityf-densityi
        mean_diff = np.mean(diff*[diff>0])
        return mean_diff

        
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

        pred1 = self.predict(uX1_L)
        pred2 = self.predict(uX2_L)
        phyloss = density_diff(density(pred1), density(pred2))
        #print("phyLoss:", 500*phyloss)
        log_likelihood -= 500*phyloss
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
        
        
        
    gp2 = GaussianProcessRegressor(kernel=pre_trained_hyperparamters, alpha =1.5, n_restarts_optimizer=10)
    gp2.fit(trainX, trainY)

    # scale the uniform numbers to original space
    # max and min value in each column 
    max_in_column_Xc = np.max(trainX,axis=0)
    min_in_column_Xc = np.min(trainX,axis=0)
        
    # Xc_scaled = (Xc-min_in_column_Xc)/(max_in_column_Xc-min_in_column_Xc)
    Xc_org = Xx*(max_in_column_Xc-min_in_column_Xc) + min_in_column_Xc
    
    samples = gp2.sample_y(Xc_org, n_samples=int(nsim)).T
    return samples