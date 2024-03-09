from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

from causallearn.utils.KCI.KCI import KCI_UInd


class M_ANM(object):

    def __init__(self, kernelX='Gaussian', kernelY='Gaussian'):

        self.kernelX = kernelX
        self.kernelY = kernelY

    def fit_gp(self, X, y, conf):
        kernel = C(1.0, (1e-10, 1e10)) * RBF(1.0, (1e-10, 1e10)) + WhiteKernel(0.1, (1e-10, 1e+10))
        gpr = GaussianProcessRegressor(kernel = kernel)

        # fit Gaussian process, including hyperparameter optimization
        X = X.join(conf)
        
        gpr.fit(X, y)
        pred_y = gpr.predict(X).reshape(-1, 1)
        return pred_y

    def cause_or_effect(self, data_x, data_y, conf):
        # set up unconditional test
        kci = KCI_UInd('Gaussian', 'Gaussian')


        # test x->y

        pred_y = self.fit_gp(data_x, data_y, conf)
        res_y = data_y - pred_y

        pval_forward, _ = kci.compute_pvalue(data_x, res_y)


        # test y->x

        pred_x = self.fit_gp(data_y, data_x, conf)
        res_x = data_x - pred_x
        pval_backward, _ = kci.compute_pvalue(data_y, res_x)


        return pval_forward, pval_backward