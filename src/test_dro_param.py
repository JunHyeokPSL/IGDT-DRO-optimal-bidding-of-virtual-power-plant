# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 03:35:55 2024

@author: HOME
"""

from scipy.optimize import minimize_scalar
import numpy as np
import time

class DRO(object):
    """
    Description: This class represents the object that calculates and provides
    access to the DRO offset.

    """

    def __init__(self, residuals: np.array):
        """
        Description: Initializes the DRO class object.
        
        Args:
            residuals (numpy.array): random samples comprising the empirical distribution
            eta (float): chance constraint risk metric
            beta (float): probability that the ambiguity set contains true distribution
            verbosity (bool): flag for printing results

        Returns:

        """
        def dim_test(residuals):
            rows, cols = residuals.shape
            # Tests correct formatting of input data:
            if rows>cols:
                residuals = residuals.T
                print('Input data incorrectly formatted - columns are random variables and rows are samples')

        self.residuals = residuals  # n x m, where n is the dimension of a sample, and
        # m is the total number of samples:
        self.n, self.m = self.residuals.shape

        self.beta = beta
        self.numels = self.residuals
        
        self.norm_dat()
        self.radius_calc()
        

    def norm_dat(self):
        """
        Description: Normalizes and centers the empirical distribution
        
        Args:

        Returns:

        """

        SIG = (self.residuals.std(axis=1))**2
        mu = np.reshape(np.mean(self.residuals, axis=1), (self.n, 1))
        self.mu = mu
        self.SIG = np.reshape(SIG, (self.n, 1))

        sigi = SIG**(-0.5)
        self.sigi = np.reshape(sigi, (self.n, 1))

        thet = np.multiply(self.residuals - self.mu, sigi)
        cut_off_eps = 1e-1

        thet[thet<cut_off_eps] = cut_off_eps

        self.thet = thet


    def radius_calc(self):
        """
        Description: Calculates the radius of the Wasserstein ambiguity set.
        
        Args:

        Returns:

        """

        def obj_c(alpha: float)-> float: 
            '''
            Objective function for radius calculation, found from:

            Chaoyue Zhao, Yongpei Guan, Data-driven risk-averse stochastic optimization 
            with Wasserstein metric, Operations Research Letters, Volume 46, Issue 2, 2018, 
            Pages 262-267, ISSN 0167-6377, https://doi.org/10.1016/j.orl.2018.01.011.

            Args: 
                alpha (float): decision variable

            Returns:
                J (float): objective function

            '''
            
            test = np.absolute(self.thet)

            J = np.sqrt(np.absolute( (1/(2*alpha))*(1+np.log(1/self.m*np.sum(np.exp(alpha*test**2)))  )))

            return J

        alphaX = minimize_scalar(obj_c, method = 'bounded', bounds = (0.001, 100))
        C = 2*alphaX.x
        Dd = 2*C

        self.epsilon = Dd*np.sqrt((2/self.m)*np.log10(1/(1-self.beta)))
        


num_samples = 150  # number of datums
random_data = np.random.normal(loc=0.0, scale=1.00, size = (1, num_samples)) * 1000
# random_data[1,:] = random_data[1,:] # +2  # optional offset
random_dat_1d = random_data[0,:]
#random_dat_1d2 = random_data[1,:]
random_data = np.abs(random_data)

cut_off_eps = 1e-3

random_data[random_data<cut_off_eps] = cut_off_eps


# DRO:
beta = 0.99  # probability that ambiguity set includes true distribution

residuals = random_data  # n x m, where n is the dimension of a sample, and
# m is the total number of samples:
n, m = residuals.shape

numels = residuals

SIG = (residuals.std(axis=1))**2
mu = np.reshape(np.mean(residuals, axis=1), (n, 1))
mu = mu
SIG = np.reshape(SIG, (n, 1))

sigi = SIG**(-0.5)
sigi = np.reshape(sigi, (n, 1))

thet = np.multiply(residuals - mu, sigi[:,np.newaxis])
DRO_object = DRO(random_data)

