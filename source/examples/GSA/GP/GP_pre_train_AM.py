import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing

def Gp_phy():

    # Physics data
    data_phyloss = np.loadtxt('../../../../data/unlabeled_data_BK_constw_v2_1525.dat')
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
    
    
    kernel = C(1.0, (1e-2, 1e1)) * RBF(length_scale = [1] * x_unlabeled.shape[1], length_scale_bounds=(1e-2, 1e9))
    gp1 = GaussianProcessRegressor(kernel=kernel, alpha =1.1, n_restarts_optimizer=15)
    gp1.fit(x_unlabeled, y_unlabeled)
    # y_pred1, sigma1 = gp1.predict(testX, return_std=True)
    # y_pred1

    pre_trained_hyperparamters = gp1.kernel_
    
    return pre_trained_hyperparamters
