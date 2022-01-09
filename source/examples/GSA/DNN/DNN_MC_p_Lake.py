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
        # K.set_session(sess)
        tf.compat.v1.keras.backend.set_session(sess)

    ss = 1
    fix_seeds(ss)

    # MC dropout
    class MCDropout(Dropout):
        def call(self, inputs, training=None):
            return super(MCDropout, self).call(inputs, training=True)

    # Compute the RMSE given the ground truth (y_true) and the predictions(y_pred)
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def PGNN_train_test(optimizer_name, optimizer_val, drop_rate, iteration, n_layers, n_nodes, tr_size, reg, lake_name):

        # Hyper-parameters of the training process
        # batch_size = int(tr_size/2)
        batch_size = 1000
        num_epochs = 1000
        val_frac = 0.2
        patience_val = 100

        # Initializing results filename
        exp_name = optimizer_name + '_drop' + str(drop_rate) + '_nL' + str(n_layers) + '_nN' + str(n_nodes) + '_trsize' + str(tr_size) + '_iter' + str(iteration)
        exp_name = exp_name.replace('.','pt')
        results_dir = '../results/'
        model_name = results_dir + exp_name + '_NoPhyInfomodel.h5' # storing the trained model
        if reg:
            results_name = results_dir + exp_name + '_results_regularizer.dat' # storing the results of the model
        else:
            results_name = results_dir + exp_name + '_results.dat' # storing the results of the model

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
            model.add(MCDropout(rate=drop_rate))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mean_squared_error',
                      optimizer=optimizer_val,
                      metrics=[root_mean_squared_error])

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience_val,verbose=1)

        print('Running...' + optimizer_name)
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

        # fix_seeds(1)

        # List of optimizers to choose from    
        optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
        optimizer_vals = [Adagrad(clipnorm=1), Adadelta(clipnorm=1), Adam(clipnorm=1), Nadam(clipnorm=1), RMSprop(clipnorm=1), SGD(clipnorm=1.), SGD(clipnorm=1, nesterov=True)]

        # selecting the optimizer
        optimizer_num = 2
        optimizer_name = optimizer_names[optimizer_num]
        optimizer_val = optimizer_vals[optimizer_num]

        # Selecting Other Hyper-parameters
        drop_rate = 0.1 # Fraction of nodes to be dropped out
        n_layers = 2 # Number of hidden layers
        n_nodes = 15 # Number of nodes per hidden layer

        # # Iterating over different training fractions and splitting indices for train-test splits
        # trsize_range = [4,6,8,10,20]

        # #default training size = 5000
        # tr_size = trsize_range[4]

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
            # results, result_file, pred, obs, rmse = PGNN_train_test(optimizer_name, optimizer_val, drop_rate, 
                            # iteration, n_layers, n_nodes, tr_size, lamda, reg)
            # testrmse.append(rmse)
            pred = PGNN_train_test(optimizer_name, optimizer_val, drop_rate, 
                            iteration, n_layers, n_nodes, tr_size, reg, lake_name)
    
    return np.squeeze(pred)
