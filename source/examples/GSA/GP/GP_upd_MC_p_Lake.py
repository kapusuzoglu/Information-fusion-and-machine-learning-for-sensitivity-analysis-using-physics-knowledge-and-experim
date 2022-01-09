import scipy.io as spio
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
def pass_arg_upd(Xx, nsim, tr_size, pre_trained_hyperparamters):

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

    ## train and test data
    #trainX, trainY = Xc[:tr_size,:-1], Y[:tr_size]
    #testX, testY = Xc[-50:,:-1], Y[-50:]

    # # Loading unsupervised data
    # unsup_filename = lake_name + '_sampled.mat'
    # unsup_mat = spio.loadmat(data_dir+unsup_filename, squeeze_me=True,
    # variable_names=['Xc_doy1','Xc_doy2'])

    # uX1 = unsup_mat['Xc_doy1'] # Xc at depth i for every pair of consecutive depth values
    # uX2 = unsup_mat['Xc_doy2'] # Xc at depth i + 1 for every pair of consecutive depth values
    # uX1 = uX1[:pre_tr_size,:-1]
    # uX2 = uX2[:pre_tr_size,:-1]
    # uY1 = uX1[:pre_tr_size,-1:]
    # uY2 = uX2[:pre_tr_size,-1:]
           
    # kernel = C(5.0, (1e-2, 1e3)) * RBF(length_scale = [1] * trainX.shape[1], length_scale_bounds=(1e-3, 1e4))
    # gp1 = GaussianProcessRegressor(kernel=kernel, alpha =1.2, n_restarts_optimizer=0)
    # gp1.fit(uX1, uY1)
    
    # # pre-trained model parameters
    # pre_trained_hyperparamters = gp1.kernel_

    # Updated model
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