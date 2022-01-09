import scipy.io as spio
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
def pass_arg_upd(nsim, tr_size, pre_trained_hyperparamters):

    print("tr_Size:",tr_size)
    #pre_tr_size = 100
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


    # Updated model
    gp2 = GaussianProcessRegressor(kernel=pre_trained_hyperparamters, alpha =.1, n_restarts_optimizer=10)
    gp2.fit(trainX, trainY)


    # scale the uniform numbers to original space
    # max and min value in each column 
    max_in_column_Xc = np.max(trainX,axis=0)
    min_in_column_Xc = np.min(trainX,axis=0)
        
    return [gp2, max_in_column_Xc, min_in_column_Xc]