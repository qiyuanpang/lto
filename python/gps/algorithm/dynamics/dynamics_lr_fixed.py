""" This file defines linear regression with an arbitrary prior. """
import numpy as np

from gps.algorithm.algorithm_utils import gauss_fit_joint_prior

class DynamicsLRFixed(object):
    """ Deterministic linear dynamics. """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        
        # Fitted dynamics: x_t+1 = Fm * [x_t;u_t] + fv.
        self.Fm = np.array([1])
        self.fv = np.array([0])
        self.dyn_covar = np.array([0])  # Covariance.
        
        self.prior = self._hyperparams['prior']['type']
                
    
    def update_prior(self, samples):
        """ Do nothing here. """

        assert self.prior == None

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior

    def fit(self, X, U):
        """ Do nothing here. """
        
        assert self.Fm == 1 and self.fv == 0 and self.dyn_covar == 0
    
    def copy(self):
        """ Return a copy of the dynamics estimate. """
        dyn = type(self)(self._hyperparams)
        dyn.Fm = np.copy(self.Fm)
        dyn.fv = np.copy(self.fv)
        dyn.dyn_covar = np.copy(self.dyn_covar)
        return dyn
