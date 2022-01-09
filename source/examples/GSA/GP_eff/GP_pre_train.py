import scipy.io as spio
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing

def Gp_phy():

    pre_tr_size = 3000
    
    #List of lakes to choose from
    lake = ['mendota' , 'mille_lacs']
    lake_num = 0  # 0 : mendota , 1 : mille_lacs
    lake_name = lake[lake_num]

    # Load features (Xc) and target values (Y)
    data_dir = '../../../../data/'
    filename = lake_name + '.mat'

    # Loading unsupervised data
    unsup_filename = lake_name + '_sampled.mat'
    unsup_mat = spio.loadmat(data_dir+unsup_filename, squeeze_me=True,
    variable_names=['Xc_doy1','Xc_doy2'])

    uX1 = unsup_mat['Xc_doy1'] # Xc at depth i for every pair of consecutive depth values
    #uX2 = unsup_mat['Xc_doy2'] # Xc at depth i + 1 for every pair of consecutive depth values
    uX1 = uX1[range(0,649723,51),:]
    scaler = preprocessing.StandardScaler()
    uX1 = scaler.fit_transform(uX1)
    
    #uX2 = uX2[range(0,649723,51),:]
    uX1_pre = uX1[:pre_tr_size,:-1]
    #uX2_pre = uX2[:pre_tr_size,:-1]
    uY1_pre = uX1[:pre_tr_size,-1:]
    #uY2_pre = uX2[:pre_tr_size,-1:]
           

    
    kernel = C(5.0, (1e-2, 1e2)) * RBF(length_scale = [1] * uX1_pre.shape[1], length_scale_bounds=(1e-1, 1e13))
    gp1 = GaussianProcessRegressor(kernel=kernel, alpha =.1, n_restarts_optimizer=15)
    gp1.fit(uX1_pre, uY1_pre)
    # y_pred1, sigma1 = gp1.predict(testX, return_std=True)
    # y_pred1

    pre_trained_hyperparamters = gp1.kernel_
    print(type(pre_trained_hyperparamters))
    return pre_trained_hyperparamters
