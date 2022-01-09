import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
def pass_arg(nsim, tr_size):

    print("tr_Size:",tr_size)
    tr_size = int(tr_size)

    # Load labeled data
    data = np.loadtxt('../../../../data/labeled_data.dat')
    x_labeled = data[:, :2].astype(np.float64) # -2 because we do not need porosity predictions
    y_labeled = data[:, -2:-1].astype(np.float64) # dimensionless bond length and porosity measurements

    # normalize dataset with MinMaxScaler
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x_labeled = scaler.fit_transform(x_labeled)
    y_labeled = scaler1.fit_transform(y_labeled)

    # train and test data
    trainX, testX, trainY, testY = train_test_split(x_labeled, y_labeled, train_size=tr_size/x_labeled.shape[0], 
                                                     random_state=42, shuffle=True)

    
    kernel = C(1.0, (1e-3, 1e1)) * RBF(length_scale = [1] * trainX.shape[1], length_scale_bounds=(1e-2, 1e7))
    gp = GaussianProcessRegressor(kernel=kernel, alpha =.1, n_restarts_optimizer=10)
    gp.fit(trainX, trainY)
    #y_pred1, sigma1 = gp.predict(testX, return_std=True)

    return gp