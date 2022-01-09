from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras import backend as K
from keras.losses import mean_squared_error
from keras.models import load_model, Model
import tensorflow as tf

from sklearn.model_selection import train_test_split
# Normalize the data.
# from sklearn import preprocessing
from keras.regularizers import l1_l2
import scipy.io as spio
import random

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

    def PGNN_train_test(optimizer_name, optimizer_val, drop_rate, iteration, n_layers, n_nodes, tr_size, pre_train, use_YPhy, lake_name):

        # Hyper-parameters of the training process
    #     batch_size = int(tr_size/2)
        batch_size = 1000
        num_epochs = 1000
        val_frac = 0.2
        patience_val = 100

        # Initializing results filename
        exp_name = "Pre-train" + optimizer_name + '_drop' + str(drop_rate) + '_nL' + str(n_layers) + '_nN' + str(n_nodes) + '_trsize' + str(tr_size) + '_iter' + str(iteration)
        exp_name = exp_name.replace('.','pt')
        results_dir = '../../../../results/Lake/'
        model_name = results_dir + exp_name + '.h5' # storing the trained model
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

        dependencies = {'root_mean_squared_error': root_mean_squared_error}

        # load the pre-trained model using non-calibrated physics-based model predictions (./data/unlabeled.dat)
        loaded_model = load_model(results_dir + pre_train, custom_objects=dependencies)

        # Creating the model
        model = Sequential()
        for layer in np.arange(n_layers):
            if layer == 0:
                model.add(Dense(n_nodes, activation='relu', input_shape=(np.shape(trainX)[1],)))
            else:
                model.add(Dense(n_nodes, activation='relu', kernel_regularizer=l1_l2(l1=.00, l2=.00)))
            # model.add(Dropout(rate=drop_rate))
            model.add(MCDropout(rate=drop_rate))
        model.add(Dense(1, activation='linear'))

        # pass the weights to all layers but 1st input layer, whose dimensions are updated
        for new_layer, layer in zip(model.layers[1:], loaded_model.layers[1:]):
            new_layer.set_weights(layer.get_weights())

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

        fix_seeds(1)

        # List of optimizers to choose from    
        optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
        optimizer_vals = [Adagrad(clipnorm=1), Adadelta(clipnorm=1), Adam(clipnorm=1), Nadam(clipnorm=1), RMSprop(clipnorm=1), SGD(clipnorm=1.), SGD(clipnorm=1, nesterov=True)]

        # selecting the optimizer
        optimizer_num = 2
        optimizer_name = optimizer_names[optimizer_num]
        optimizer_val = optimizer_vals[optimizer_num]

        # Selecting Other Hyper-parameters
        drop_rate = 0.1 # Fraction of nodes to be dropped out
        use_YPhy = 0 # Whether YPhy is used as another feature in the NN model or not
        n_layers = 2 # Number of hidden layers
        n_nodes = 15 # Number of nodes per hidden layer
        
        # pre-trained model
        pre_train = 'Scaled_Lake_Pre-trainAdam_drop0pt1_nL2_nN15_trsize600000_iter0.h5'
        tr_size = int(tr_size)

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
                        iteration, n_layers, n_nodes, tr_size, pre_train, use_YPhy, lake_name)
    
    return np.squeeze(pred)
