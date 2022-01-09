import scipy.io as spio
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
def pass_arg_upd(Xx, nsim, tr_size, pre_trained_hyperparamters):

    print("tr_Size:",tr_size)
    #pre_tr_size = 100
    tr_size = int(tr_size)

    # Load labeled data
    data = np.loadtxt('../../../../data/labeled_data.dat')
    x_labeled = data[:, :2].astype(np.float64) # -2 because we do not need porosity predictions
    y_labeled = data[:, -2:-1].astype(np.float64) # dimensionless bond length and porosity measurements

    # normalize dataset with MinMaxScaler
    scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
    x_labeled = scaler.fit_transform(x_labeled)
    y_labeled = scaler.fit_transform(y_labeled)

    # train and test data
    trainX, testX, trainY, testY = train_test_split(x_labeled, y_labeled,train_size=tr_size/x_labeled.shape[0], 
                                                     random_state=42, shuffle=True)


    # Updated model
    gp2 = GaussianProcessRegressor(kernel=pre_trained_hyperparamters, alpha =.1, n_restarts_optimizer=10)
    gp2.fit(trainX, trainY)
    
    samples = gp2.sample_y(Xx, n_samples=int(nsim)).T
    return np.squeeze(samples)