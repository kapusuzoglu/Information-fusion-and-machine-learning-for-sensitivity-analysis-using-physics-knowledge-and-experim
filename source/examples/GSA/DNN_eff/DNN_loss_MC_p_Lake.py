from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras import backend as K
from keras.losses import mean_squared_error
import tensorflow as tf

from sklearn.model_selection import train_test_split
# Normalize the data.
from keras.regularizers import l1_l2

import random
import scipy.io as spio

def pass_arg(Xx, nsim, tr_size):
    print("Tr_size:", tr_size)
    def fix_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #     K.set_session(sess)
        tf.compat.v1.keras.backend.set_session(sess)

    ss = 1
    fix_seeds(ss)

    # MC dropout
    class MCDropout(Dropout):
        def call(self, inputs, training=None):
            return super(MCDropout, self).call(inputs, training=True)

    # import pickle

    # def save_obj(obj, name):
    #     with open(name, 'wb') as f:
    #         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    
    # Compute the RMSE given the ground truth (y_true) and the predictions(y_pred)
    def root_mean_squared_error(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


    #function for computing the density given the temperature(nx1 matrix)
    def density(temp):
        return 1000 * ( 1 - (temp + 288.9414) * (temp - 3.9863)**2 / (508929.2 * (temp + 68.12963) ) )

    def phy_loss_mean(params):
        # useful for cross-checking training
        udendiff, lam = params
        def loss(y_true,y_pred):
            return K.mean(K.relu(udendiff))
        return loss

    #function to calculate the combined loss = sum of rmse and phy based loss
    def combined_loss(params):
        udendiff, lam = params
        def loss(y_true,y_pred):
            return mean_squared_error(y_true, y_pred) + lam * K.mean(K.relu(udendiff))
        return loss

    def PGNN_train_test(optimizer_name, optimizer_val, drop_frac, use_YPhy, iteration, n_layers, n_nodes, tr_size, lamda, reg, lake_num):

    #     fix_seeds(ss)

        # Hyper-parameters of the training process
    #     batch_size = tr_size
        batch_size = 1000
        num_epochs = 1000
        val_frac = 0.2
        patience_val = 100

        # Initializing results filename
        exp_name = "DNN_loss" + optimizer_name + '_drop' + str(drop_frac) + '_usePhy' + str(use_YPhy) +  '_nL' + str(n_layers) + '_nN' + str(n_nodes) + '_trsize' + str(tr_size) + '_lamda' + str(lamda) + '_iter' + str(iteration)
        exp_name = exp_name.replace('.','pt')
        results_dir = '../results/'
        model_name = results_dir + exp_name + '_model.h5' # storing the trained model
        
        # Load features (Xc) and target values (Y)
        data_dir = '../../../../data/'
        filename = lake_name + '.mat'
        mat = spio.loadmat(data_dir + filename, squeeze_me=True,
        variable_names=['Y','Xc_doy','Modeled_temp'])
        Xc = mat['Xc_doy']
        Y = mat['Y']
        Xc = Xc[:,:-1] # remove Y_phy, physics model outputs
        # train and test data
        trainX, testX, trainY, testY = train_test_split(Xc, Y, train_size=tr_size/Xc.shape[0], 
                                                    test_size=tr_size/Xc.shape[0], random_state=42, shuffle=True)

        ## train and test data
        #trainX, trainY = Xc[:tr_size,:], Y[:tr_size]
        #testX, testY = Xc[-50:,:], Y[-50:]

        
        # Loading unsupervised data
        unsup_filename = lake_name + '_sampled.mat'
        unsup_mat = spio.loadmat(data_dir+unsup_filename, squeeze_me=True,
        variable_names=['Xc_doy1','Xc_doy2'])

        uX1 = unsup_mat['Xc_doy1'] # Xc at depth i for every pair of consecutive depth values
        uX2 = unsup_mat['Xc_doy2'] # Xc at depth i + 1 for every pair of consecutive depth values
        #uX1 = uX1[:50000,:]
        #uX2 = uX2[:50000,:]
        uX1 = uX1[range(0,649723,51),:]
        uX2 = uX2[range(0,649723,51),:]
        
        if use_YPhy == 0:
            # Removing the last column from uX (corresponding to Y_PHY)
            uX1 = uX1[:,:-1]
            uX2 = uX2[:,:-1]
        # Creating the model
        model = Sequential()
        for layer in np.arange(n_layers):
            if layer == 0:
                model.add(Dense(n_nodes, activation='relu', input_shape=(np.shape(trainX)[1],)))
            else:
                if reg:
                    model.add(Dense(n_nodes, activation='relu', kernel_regularizer=l1_l2(l1=.00, l2=.00)))
                else:
                    model.add(Dense(n_nodes, activation='relu'))
            # model.add(Dropout(rate=drop_frac))
            model.add(MCDropout(rate=drop_frac))
        model.add(Dense(1, activation='linear'))


        # physics-based regularization
        uin1 = K.constant(value=uX1) # input at depth i
        uin2 = K.constant(value=uX2) # input at depth i + 1
        lam = K.constant(value=lamda) # regularization hyper-parameter
        uout1 = model(uin1) # model output at depth i
        uout2 = model(uin2) # model output at depth i + 1
        udendiff = (density(uout1) - density(uout2)) # difference in density estimates at every pair of depth values

        totloss = combined_loss([udendiff, lam])
        phyloss = phy_loss_mean([udendiff, lam])

        model.compile(loss=totloss,
                      optimizer=optimizer_val,
                      metrics=[phyloss, root_mean_squared_error])

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience_val, verbose=1)

    #     print('Running...' + optimizer_name)
        history = model.fit(trainX, trainY,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            verbose=0,
                            validation_split=val_frac, callbacks=[early_stopping, TerminateOnNaN()])
    
        test_score = model.evaluate(testX, testY, verbose=1)
        print(test_score)

        # scale the uniform numbers to original space
        # max and min value in each column 
        max_in_column_Xc = np.max(trainX,axis=0)
        min_in_column_Xc = np.min(trainX,axis=0)
        
        # Xc_scaled = (Xc-min_in_column_Xc)/(max_in_column_Xc-min_in_column_Xc)
        Xc_org = Xx*(max_in_column_Xc-min_in_column_Xc) + min_in_column_Xc
        
        samples = []
        for i in range(int(nsim)):
            #print("simulation num:",i)
            predictions = model.predict(Xc_org)
            samples.append(predictions)
        return np.array(samples)
    
    



    # Main Function
    if __name__ == '__main__':

        fix_seeds(1)

        # List of optimizers to choose from    
        optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
        optimizer_vals = [Adagrad(clipnorm=1), Adadelta(clipnorm=1), Adam(clipnorm=1), Nadam(clipnorm=1), RMSprop(clipnorm=1), SGD(clipnorm=1.), SGD(clipnorm=1, nesterov=True)]

        # selecting the optimizer
        optimizer_num = 2
        optimizer_name = optimizer_names[optimizer_num]
        optimizer_val = optimizer_vals[optimizer_num]

        # Selecting Other Hyper-parameters
        drop_frac = 0.1 # Fraction of nodes to be dropped out
        use_YPhy = 0 # Whether YPhy is used as another feature in the NN model or not
        n_layers = 2 # Number of hidden layers
        n_nodes = 15 # Number of nodes per hidden layer

        #set lamda
        lamda = 10 # Physics-based regularization constant  
        
        tr_size = int(tr_size)

        # use regularizer
        reg = True

        # total number of runs
        iter_range = np.arange(1)
        
        
        #List of lakes to choose from
        lake = ['mendota' , 'mille_lacs']
        lake_num = 0  # 0 : mendota , 1 : mille_lacs
        lake_name = lake[lake_num]
        
        testrmse=[]
        # iterating through all possible params
        for iteration in iter_range:
#             results, result_file, pred, obs, rmse, obs_train = PGNN_train_test(optimizer_name, optimizer_val, drop_frac, use_YPhy, 
#                             iteration, n_layers, n_nodes, tr_size, lamda, reg, samp)
#             testrmse.append(rmse)
            #print(iteration)
            pred = PGNN_train_test(optimizer_name, optimizer_val, drop_frac, use_YPhy, 
                            iteration, n_layers, n_nodes, tr_size, lamda, reg, lake_num)
            

    return np.squeeze(pred)