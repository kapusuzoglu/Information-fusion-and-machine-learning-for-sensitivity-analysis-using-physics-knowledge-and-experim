import numpy as np
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#from sklearn.gaussian_process import GaussianProcessRegressor

def pass_arg2(Xx, nsim, gp):
        
    print(gp.kernel_)
    samples = gp.sample_y(Xx, n_samples=int(nsim)).T
    print(np.squeeze(samples).shape)
    return np.squeeze(samples)