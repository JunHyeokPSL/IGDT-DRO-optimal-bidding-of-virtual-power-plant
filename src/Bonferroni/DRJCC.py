# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 20:53:59 2023

@author: junhyeok

#Reference: Github @chrord Energy_and_reserves_dispatch_with_DRJCC

# Bonferroni / DRO_CVaR_ICC 

"""

import os,sys
from scipy import io
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB 
import time

import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})


BASE_DIR = os.getcwd()
os.chdir("../")
path = os.getcwd() 
sys.path.append(path) # 폴더 한 단계 위에서 file import 하기 위해서 sys path 설정

# Getting the number of thermals power plants , wind farmds, solar panels, scenarios.

Nunits = 100
Nwind = 100
Npv = 100
Nscen = 100

# Definition of variables


class system_info:
    
    def __init__(scale_factor):

        BASE_DIR = os.getcwd()
        path = BASE_DIR
        mat_file = io.loadmat(f'{path}/Data_Generation/AV_AEMO')
        
        self.scale_factor = scale_factor
        
        
        
        
        
class DRO_CVaR_ICC:
    
    def __init__(self, si, DRO_param, jcc):
        
        
        self.si = si
        self.DRO_param = DRO_param
        self.jcc = jcc
        
        
        self.nPV = si.nPV
        self.nWT = si.nWT
        self.nScen = si.nScen
        
        
        
        
        
        
        
class DRO_param:
    def __init__(self):

        self.dual_norm = 'inf'; # dual norm
        self.eps_joint_cvar = 0.05; # \epsilon
        self.CVaR_max_iter = 40; # MaxIter
        self.tolerance = 1e-1; # \eta
        self.alpha_max = 1000; # \overline{\delta}
        self.alpha_min = 1e-4; # \undeline{\delta}
        self.penalty = 1e6; # BigM

    def set_eps_joint_cvar(self, jcvar):
        self.eps_joint_cvar = jcvar        
        
if __name__ == '__main__':
    
    BASE_DIR = os.getcwd()
    path = BASE_DIR
    mat_file = io.loadmat(f'{path}/Data_Generation/AV_AEMO')
    
    
    