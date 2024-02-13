# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:30:59 2023

@author: user
"""

import numpy as np
from scipy import io
import pandas as pd

def generate_gen_dict(num, dg_max, model_dict):
    
    gen_dict = {}
    if num == 0:
        a = 94.5487
        gen_dict = {'min_power': dg_max * 0.1,
               'max_power': dg_max ,
               'ramp_up_limit': dg_max * 0.3,
               'ramp_down_limit': dg_max * 0.3 ,
               '상수': a,
               '1차 계수': 2.21,
               '2차 계수': 0.00033,
               'start_up_cost': a,
               'shut_down_cost': a,
               'fuel_cost': 55.117,
               'N_PIECE': model_dict['N_PIECE'],
               }    
    elif num == 1:
        a = 95.343
        gen_dict = {'min_power': dg_max * 0.1,
               'max_power': dg_max ,
               'ramp_up_limit': dg_max * 0.4,
               'ramp_down_limit': dg_max * 0.4 ,
               '상수': a,
               '1차 계수': 2.15,
               '2차 계수': 0.00037,
               'start_up_cost': a,
               'shut_down_cost': a,
               'fuel_cost': 55.128,
               'N_PIECE': model_dict['N_PIECE'],
               }        
    return gen_dict

def generate_smp(path):
    
    da_smp_dataset = pd.read_excel(f"{path}/src/Data_Generation/smp_jeju_2022.xlsx", header=1)
    da_smp_data = da_smp_dataset.loc[da_smp_dataset['구분']== 20221030]
    da_smp_profile= da_smp_data.iloc[0,1:1+24].values
    da_smp_profile = da_smp_profile.astype(float)
    
    return da_smp_profile
    


def PTDF_calc(N_El_nodes, ref_node, ElNetwork, system_info):
    
    # ADMITTANCE MATRIX
    B_N = np.zeros((N_El_nodes, N_El_nodes))
    B_L = np.zeros((ElNetwork.shape[0], N_El_nodes))

    # Off-diagonal elements B-matrix
    for l in range(ElNetwork.shape[0]):  # Number of Lines

        From = int(ElNetwork[l,0] - 100 -1)
        To = int(ElNetwork[l,1] - 100 -1) 
        X = ElNetwork[l,2]
        
        B_N[From , To] = -1 / X
        B_N[To, From] = -1 / X

        B_L[l, From] = 1 / X
        B_L[l, To] = -1 / X
        
    # Diagonal elements B-matrix
    for k in range(N_El_nodes):
        B_N[k, k] = -np.sum(B_N[k, :])

    ref_node = ref_node - 1 #matlab to python -1
    # Remove ref node
    B_NN = np.delete(np.delete(B_N, ref_node, axis=0), ref_node, axis=1)
    B_LL = np.delete(B_L, ref_node, axis=1)

    PTDF_nrf = np.dot(B_LL, np.linalg.inv(B_NN))

    PTDF = np.hstack((np.round(PTDF_nrf[:, :ref_node], 2), np.zeros((PTDF_nrf.shape[0], 1)),
                  np.round(PTDF_nrf[:, ref_node:], 2)))
    
    system_info['B_N'] = B_N
    system_info['B_L'] = B_L
    
    system_info['B_NN'] = B_NN
    system_info['B_LL'] = B_LL
    
    system_info['PTDF_nrf'] = PTDF_nrf
    system_info['PTDF'] = PTDF
    
    return PTDF, system_info
    
def generate_wind(path, wind_dict):
    
    nWT = wind_dict['nWT']
    nScen = wind_dict['nScen']
    N_max = wind_dict['N_max']
    OOS_max = wind_dict['OOS_max']
    IR_max = wind_dict['IR_max']
    
    N = wind_dict['N']
    OOS_sim = wind_dict['OOS_sim']
    
    wt_profile_dict = io.loadmat(f'{path}/src/Data_Generation/AV_AEMO')
    wff = wt_profile_dict['AV_AEMO2'][:, :nWT]

    # Cutting off very extreme values

    cut_off_eps = 1e-2
    wff[wff<cut_off_eps] = cut_off_eps;
    wff[wff>(1-cut_off_eps)] = 1 - cut_off_eps;

    # Logit-normal transformation (Eq. (1) in ref. [31])
    yy = np.log(wff/(1-wff))

    # Calculation of mean and variance, note that we increase the mean to have
    # higher wind penetration in our test-case
    mu = yy.mean(axis=0) + 1.5 # Increase 1.5 for higher wind penetration
    cov_m = np.cov(yy, rowvar = False)
    std_yy = yy.std(axis=0).reshape(1, yy.shape[1])
    std_yy_T = std_yy.T
    sigma_m = cov_m / (std_yy_T @ std_yy)

    # Inverse of logit-normal transformation (Eq. (2) in ref. [31]) 
    R = np.linalg.cholesky(sigma_m).T

    wt_rand_pattern = np.random.randn(nScen, nWT)

    y = np.tile(mu, (nScen,1)) + wt_rand_pattern @ R
    Wind = (1 + np.exp(-y))**-1

    # Checking correlation, mean and true mean of data
    corr_check_coeff = np.corrcoef(Wind, rowvar = False)
    mu_Wind = Wind.mean(axis=0)
    true_mean_Wind = (1+ np.exp(-mu))**-1

    # Reshaping the data structure


    nWind = Wind.T
    nWind = nWind.reshape(nWT, N_max + OOS_max, IR_max)


    # peak N and N' samples
    j = 0
    WPf_max = nWind[:,0:N_max,j].transpose()
    WPr_max = nWind[:,N_max:N_max + OOS_max, j].transpose()
    WPf = WPf_max[0:N,:]
    WPr = WPr_max[0:OOS_sim,:]

    Wscen = WPf[0:N,:].transpose()
    Wscen_mu = Wscen.mean(axis = 1)
    Wscen_mu = Wscen_mu.reshape(len(Wscen_mu),1)
    Wscen_xi = Wscen - np.tile(Wscen_mu,(1, Wscen.shape[1]))

    return Wscen, Wscen_mu, Wscen_xi

def generate_sys():
    
    ElNetwork= np.array([
        [101,     102,   0.0146,    175], #l1
        [101,     103,   0.2253,    175], #l2
        [101,     105,   0.0907,    400], #l3
        [102,     104,   0.1356,    175], #l4
        [102,     106,    0.205,    175], #l5
        [103,     109,   0.1271,    400], #l6
        [103,     124,    0.084,    200], #l7
        [104,     109,    0.111,    175], #l8
        [105,     110,    0.094,    400], #l9
        [106,     110,   0.0642,    400], #l10
        [107,     108,   0.0652,    600], #l11
        [108,     109,   0.1762,    175], #l12
        [108,     110,   0.1762,    175], #l13
        [109,     111,    0.084,    200], #l14
        [109,     112,    0.084,    200], #l15
        [110,     111,    0.084,    200], #l16
        [110,     112,    0.084,    200], #l17
        [111,     113,   0.0488,    500], #l18
        [111,     114,   0.0426,    500], #l19
        [112,     113,   0.0488,    500], #l20
        [112,     123,   0.0985,    500], #l21
        [113,     123,   0.0884,    500], #l22
        [114,     116,   0.0594,    1000], #l23
        [115,     116,   0.0172,    500], #l24
        [115,     121,   0.0249,    1000], #l25
        [115,     124,   0.0529,    500], #l26
        [116,     117,   0.0263,    500], #l27
        [116,     119,   0.0234,    500], #l28
        [117,     118,   0.0143,    500], #l29
        [117,     122,   0.1069,    500], #l30
        [118,     121,   0.0132,    1000], #l31
        [119,     120,   0.0203,    1000], #l32
        [120,     123,   0.0112,    1000], #l33
        [121,     122,   0.0692,    500] #l34
        ])


    GenDATA = np.array([
        [0, 152, 40, 40, 1, 12, 12.65, 1],    # 1
        [0, 152, 40, 40, 2, 13, 13.45, 3],    # 2
        [0, 300, 70, 70, 7, 11, 0, 0],       # 3
        [0, 591, 60, 60, 13, 17, 0, 0],      # 4
        [0, 60, 30, 30, 15, 18, 11.12, 2],   # 5
        [0, 155, 30, 30, 15, 14, 0, 0],      # 6
        [0, 155, 30, 30, 16, 15, 14.88, 1],  # 7
        [0, 400, 50, 50, 18, 5, 0, 0],       # 8
        [0, 400, 50, 50, 21, 7, 0, 0],       # 9
        [0, 300, 50, 50, 22, 20, 0, 0],      # 10
        [0, 310, 60, 60, 23, 10.52, 16.80, 2],  # 11
        [0, 350, 40, 40, 23, 10.89, 15.60, 3],  # 12
    ])

    PipeCap = [[10000],[5500],[7000]]

    DemandDATA = np.array([
        [0.038, 1, 1000],   # d1
        [0.034, 2, 1000],   # d2
        [0.063, 3, 1000],   # d3
        [0.026, 4, 1000],   # d4
        [0.025, 5, 1000],   # d5
        [0.048, 6, 1000],   # d6
        [0.044, 7, 1000],   # d7
        [0.060, 8, 1000],   # d8
        [0.061, 9, 1000],   # d9
        [0.068, 10, 1000],  # d10
        [0.093, 13, 1000],  # d11
        [0.068, 14, 1000],  # d12
        [0.111, 15, 1000],  # d13
        [0.035, 16, 1000],  # d14
        [0.117, 18, 1000],  # d15
        [0.064, 19, 1000],  # d16
        [0.045, 20, 1000],  # d17
    ])

    WindDATA = np.array([
        [0, 200, 1],   # 1
        [0, 200, 2],   # 2
        [0, 200, 11],   # 3
        [0, 200, 12],   # 4
        [0, 200, 12],   # 5
        [0, 200, 16],   # 6
    ])
    
    
    C = np.array([17.5, 20, 15, 27.5, 30, 22.5, 25, 5, 7.5, 32.5, 10, 12.5])
    Cr1 = np.array([3.5, 4, 3, 5.5, 6, 4.5, 5, 1, 1.5, 6.5, 2, 2.5])
    Cr2 = np.array([3.5, 4, 3, 5.5, 6, 4.5, 5, 1, 1.5, 6.5, 2, 2.5])
    
    Scale_Factor = 1
    Total_Demand = 2650
    

    system_info = {}
    system_info['F'] = ElNetwork[:,3] / Scale_Factor
    system_info['D'] = Total_Demand * DemandDATA[:,0] / Scale_Factor
    system_info['Pmax'] = GenDATA[:,1] / Scale_Factor
    system_info['Pmin'] = GenDATA[:,0] / Scale_Factor
    system_info['R'] = (system_info['Pmax'] + system_info['Pmin'])/2
    system_info['ResCap'] = GenDATA[:,1] * 0.4
    
    system_info['C'] = C
    system_info['Cr1'] = Cr1
    system_info['Cr2'] = Cr2
    
    system_info['Wmax'] = WindDATA[:,1] / Scale_Factor
    system_info['DWmax'] = np.diag(system_info['Wmax'])
    system_info['Wmin'] = WindDATA[:,0] / Scale_Factor
    
    
    nUnits = GenDATA.shape[0]
    nWT = WindDATA.shape[0]
    nD = DemandDATA.shape[0]
    nF = ElNetwork.shape[0]
    
    system_info['nUnits'] = nUnits
    system_info['nWT'] = nWT
    system_info['nD'] = nD
    system_info['nF'] = nF
    
    
    N_El_nodes = 24
    ref_node = 13    
    
    system_info['AG'] = np.zeros([N_El_nodes, nUnits])
    system_info['AW'] = np.zeros([N_El_nodes, nWT])
    system_info['AD'] = np.zeros([N_El_nodes, nD])
    
    for n in range(N_El_nodes):
        for gg in range(nUnits):
            if GenDATA[gg, 4]-1 == n :
                system_info['AG'][n,gg] = 1 
    
        for ww in range(nWT):
            if WindDATA[ww,2]-1 == n:
                system_info['AW'][n,ww] = 1
                
        for dd in range(nD):
            if DemandDATA[dd,1]-1 == n:
                system_info['AD'][n,dd] = 1
        
       
    PTDF, system_info = PTDF_calc(N_El_nodes, ref_node,ElNetwork, system_info)
    try:
        system_info['qG'] = PTDF @ system_info['AG']
        system_info['qW'] = PTDF @ system_info['AW']
        system_info['qD'] = PTDF @ system_info['AD']
    except Exception as error:
        print(error)            
    return system_info

def generate_matrix(system_info):
    
    jcc = []
    nUnits = system_info['nUnits']
    nWT = system_info['nWT']
    nF = system_info['nF']
    
    zG = np.zeros(nUnits)
    zGzG = np.zeros([nUnits,nUnits])
    eG = np.eye(nUnits)
    zGzW = np.zeros([nUnits,nWT])
    
    
    # Matrices for generation
    
    A_gg = np.concatenate((eG,-eG), axis=0)
    A_gw = np.concatenate([zGzW,zGzW], axis=0)
    b_gc = np.concatenate([zG,zG],axis=0)
    b_gg = np.concatenate((np.concatenate((zGzG, eG, zGzG), axis=1), np.concatenate((zGzG, zGzG, eG), axis=1)), axis=0)
    
    
    # Matrices for transmission lines
    
    qG = system_info['qG']
    qW = system_info['qW']
    DWmax = system_info['DWmax']
    F = system_info['F']
    qD = system_info['qD']
    D = system_info['D']
    mu = system_info['Wscen_mu'].flatten()
    
    
    zFzG = np.zeros([nF,nUnits])
    
    A_lg = np.concatenate((qG, -qG), axis = 0)
    A_lw = np.concatenate((qW @ DWmax, -qW@DWmax), axis = 0)
    b_lc = np.concatenate((F - qW@DWmax@mu + qD@D, F + qW@DWmax@mu - qD@D ), axis =0)
    b_lg = np.concatenate((np.concatenate((-qG, zFzG, zFzG), axis=1), np.concatenate((qG, zFzG, zFzG), axis=1)), axis=0)

    # A_g : B(j,:) == jcc{j,2} == jcc[j,0]
    # A_w : C(j,:) == jcc{j,3} == jcc[j,1]
    # b_g : A(j,:) == -jcc{j,1} == jcc[j,3]
    # b_c : b(j) == jcc{j,4} == jcc[j,2]
    
    jcc.append([A_gg, A_gw, b_gc, b_gg])
    jcc.append([A_lg, A_lw, b_lc, b_lg])
    
    return jcc

def generate_H_matrix(system_info):
    
    nWT = system_info['nWT']
    
    mu = system_info['Wscen_mu']
    Wmax = system_info['Wmax']
    eW = np.eye(nWT)
    H = np.concatenate((eW, -eW), axis = 0)
    h = np.concatenate((Wmax - mu, mu), axis = 0)    
    
    hc = [H, h]
    return hc


