import numpy as np
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#from sklearn.gaussian_process import GaussianProcessRegressor

def pass_arg2(Xx, nsim, gp_):
        
    gp = gp_[0]
    max_in_column_Xc = gp_[1]
    min_in_column_Xc = gp_[2]
    
    # Xc_scaled = (Xc-min_in_column_Xc)/(max_in_column_Xc-min_in_column_Xc)
    Xc_org = Xx*(max_in_column_Xc-min_in_column_Xc) + min_in_column_Xc
        
    print("MCdrop:",gp.kernel_)
    samples = gp.sample_y(Xc_org, n_samples=int(nsim)).T

    return np.squeeze(samples)