import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

# Normalize the data.
from sklearn import preprocessing

import random

def pass_arg(Xx, nsim, tr_size, num_iter):
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

    # Compute the RMSE given the ground truth (y_true) and the predictions(y_pred)
    def root_mean_squared_error(y_true, y_pred):
            return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true), axis=-1))

    class InputTransformedKernel(tfp.math.psd_kernels.PositiveSemidefiniteKernel):

        def __init__(self, kernel, transformation, name='InputTransformedKernel'):
            self._kernel = kernel
            self._transformation = transformation
            super(InputTransformedKernel, self).__init__(
                feature_ndims=kernel.feature_ndims,
                dtype=kernel.dtype,
                name=name)

        def apply(self, x1, x2):
            return self._kernel.apply(
                self._transformation(x1),
                self._transformation(x2))

        def matrix(self, x1, x2):
            return self._kernel.matrix(
                self._transformation(x1),
                self._transformation(x2))

        @property
        def batch_shape(self):
            return self._kernel.batch_shape

        def batch_shape_tensor(self):
            return self._kernel.batch_shape_tensor

    class InputScaledKernel(InputTransformedKernel):

        def __init__(self, kernel, length_scales):
            super(InputScaledKernel, self).__init__(
                kernel,
                lambda x: x / tf.expand_dims(length_scales,
                                         -(kernel.feature_ndims + 1)))

    # Load labeled data
    data = np.loadtxt('../data/labeled_data.dat')
    x_labeled = data[:, :2].astype(np.float64) # -2 because we do not need porosity predictions
    y_labeled = data[:, -2:-1].astype(np.float64) # dimensionless bond length and porosity measurements

    # normalize dataset with MinMaxScaler
    scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
    x_labeled = scaler.fit_transform(x_labeled)
    # y_labeled = scaler.fit_transform(y_labeled)

    tr_size = int(tr_size)

    # train and test data
    trainX, trainY = x_labeled[:tr_size,:], y_labeled[:tr_size]
    # testX, testY = x_labeled[tr_size:,:], y_labeled[tr_size:]

    trainY = np.transpose(trainY)
    # testY = np.transpose(testY)


    def build_gp(amplitude, length_scale):
        """Defines the conditional dist. of GP outputs, given kernel parameters."""

        # Create the covariance kernel, which will be shared between the prior (which we
        # use for maximum likelihood training) and the posterior (which we use for
        # posterior predictive sampling)    
        se_kernel = tfk.ExponentiatedQuadratic(amplitude)  # length_scale = None here, implicitly

        # This is the "ARD" kernel (we don't like abbreviations or bizarrely obscure names in
        # TFP, so we're probably going to call this "InputScaledKernel" since....that's what it is! :)
        kernel = InputScaledKernel(se_kernel, length_scale)

        # Create the GP prior distribution, which we will use to train the model
        # parameters.
        return tfd.GaussianProcess(kernel=kernel,index_points=trainX)

    gp_joint_model = tfd.JointDistributionNamedAutoBatched({
        'amplitude': tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=0., scale=np.float64(1.)),
                bijector=tfb.Exp(),
                batch_shape=[1]),
        'length_scale': tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=0., scale=np.float64(1.)),
                bijector=tfb.Exp(),
                batch_shape=[2]),
        'observations': build_gp,
    })



    # Create the trainable model parameters, which we'll subsequently optimize.
    # Note that we constrain them to be strictly positive.
    constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    amplitude_var = tfp.util.TransformedVariable(
        initial_value=np.random.uniform(size=1),
        bijector=constrain_positive,
        name='amplitude',
        dtype=np.float64)

    length_scale_var = tfp.util.TransformedVariable(
        initial_value=np.random.uniform(size=[2]),
        bijector=constrain_positive,
        name='length_scale',
        dtype=np.float64)

    trainable_variables = [v.trainable_variables[0] for v in 
                           [amplitude_var,
                           length_scale_var]]



    @tf.function(autograph=False, experimental_compile=False)
    def target_log_prob(amplitude, length_scale):
        return - gp_joint_model.log_prob({
          'amplitude': amplitude,
          'length_scale': length_scale,
          'observations': trainY
      })


    fix_seeds(1)

    # Optimize the model parameters.
    num_iters = int(num_iter)
    optimizer = tf.optimizers.Adam(learning_rate=.1)

    # Store the likelihood values during training, so we can plot the progress
    lls_ = np.zeros(num_iters, np.float64)

    for i in range(num_iters):
        with tf.GradientTape() as tape:
            loss = target_log_prob(amplitude_var, length_scale_var)

        # print(i,"loss_inloop:",loss)
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        lls_[i] = loss

    # print('Trained parameters:')
    # print('amplitude: {}'.format(amplitude_var._value().numpy()))
    # print('length_scale: {}'.format(length_scale_var._value().numpy()))



    # tf.random.set_seed(1234)
    fix_seeds(1)
    se_kernel = tfk.ExponentiatedQuadratic(amplitude_var)  # length_scale = None here, implicitly
    optimized_kernel = InputScaledKernel(se_kernel, length_scale_var)
    gprm = tfd.GaussianProcessRegressionModel(kernel=optimized_kernel, index_points = Xx)
    preds = gprm.sample(int(nsim))
    samples = np.array(tf.squeeze(preds, axis=1))

    return samples