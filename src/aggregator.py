# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:03:24 2023

@author: junhyeok
"""

import numpy as np
import pandas as pd
from scipy import io
import scipy.stats

# Define the aggrergator class

class aggregator:
    def __init__(self, agg, char_ess, model_dict, case_dict):
        
        self.name = agg['name']
        self.code = agg['code']
        self.ngen_dict = agg['gen']
        self.nTotalGen = 0
        
        # self.wt_profile = agg['profile'][0]
        # self.pv_profile = agg['profile'][1]
        
        self.char_ess = char_ess
        
        self.wt_list = []
        self.pv_list = []
        self.ess_list = []
        self.dg_list = []
        
        self.model_dict = model_dict
        self.path = model_dict['path']
        self.nTimeslot = self.model_dict['nTimeslot']
        self.N_PIECE = model_dict['N_PIECE']
        
        self.dg_dict_list = model_dict['dg_dict_list']
        
        self.case_dict = case_dict
        
        self.divide_factor = self.case_dict['divide_factor']

            
            
        self.initialize_res()
        # self.set_smp_data()
        
    def set_smp_data(self, date=None):
        da_smp_dataset = pd.read_excel(f"{self.path}/src/Data_Generation/smp_jeju_2022.xlsx", header=1)
        
        if date is not None:
            da_smp_data = da_smp_dataset.loc[da_smp_dataset['구분'] == int(date)]
        else:
            da_smp_data = da_smp_dataset.loc[da_smp_dataset['구분'] == 20220911]
        
        da_smp_profile = da_smp_data.iloc[0, 1:1 + 24].values
        da_smp_profile = da_smp_profile.astype(float)
        
        self.da_smp_profile = da_smp_profile
        
        
    def initialize_res(self):
        
        self.nWT = self.ngen_dict['WT']
        self.nPV = self.ngen_dict['PV']
        self.nESS = self.ngen_dict['ESS']
        self.nDG= self.ngen_dict['DG']
        self.nRES = self.nWT + self.nPV
        self.nTotalGen = self.nPV + self.nWT + self.nESS + self.nDG
        
        count = 1
        
        for i in range(self.nWT):
            self.wt_list.append(WT(self.name, self.code, count))
            self.wt_list[i].set_wt_profile(self.case_dict, self.model_dict)
            count = count + 1
        
        for i in range(self.nPV):
            self.pv_list.append(PV(self.name, self.code, count))
            self.pv_list[i].set_pv_profile(self.case_dict, self.model_dict)
            count = count + 1
            
        for i in range(self.nESS):
            self.ess_list.append(ESS(self.name, self.code,count, self.char_ess))
            count = count + 1
        for i in range(self.nDG):
            self.dg_list.append(DG(self.name, self.code, count, self.dg_dict_list[i]))
            count = count + 1
            
        self.res_list = self.wt_list + self.pv_list
            
    def set_wt_power(self, max_power_list):
        
        for i in range(len(self.wt_list)):
            self.wt_list[i].set_power(max_power_list[i])
        
    def set_pv_power(self, max_power_list):
        for i in range(len(self.pv_list)):
            self.pv_list[i].set_power(max_power_list[i])
    
    def set_ess_power(self, max_power_list):
        for i in range(len(self.ess_list)):
            self.ess_list[i].set_power(max_power_list[i])
            
    def set_ess_capacity(self,max_capacity_list):
        for i in range(len(self.ess_list)):
            self.ess_list[i].set_capacity(max_capacity_list[i])

    def set_dg_spec(self, max_power_list):
        ramp_list = self.dg_dict['ramp']
        
        for i in range(len(self.dg_list)):
            self.dg_list[i].set_power(max_power_list[i])
            self.dg_list[i].set_ramp_power(ramp_list[i])
            self.dg_list[i].set_slope(self.dg_dict)
            
            
    def set_der_power(self, max_list):
        self.set_wt_power(max_list[0])
        self.set_pv_power(max_list[1])
        self.set_ess_power(max_list[2])
        self.set_ess_capacity(max_list[3])
        #self.set_dg_spec(max_list[4])
        self.set_res_table()
               
    
    def set_res_table(self):
        
        self.data_list = []
        self.total_max_power = np.zeros(self.nTimeslot) 
        self.total_min_power = np.zeros(self.nTimeslot)
        self.gen_list = [self.wt_list, self.pv_list, self.ess_list, self.dg_list]
        for i in range(len(self.gen_list)):
            for j in range(len(self.gen_list[i])):
                self.data_list.append(self.gen_list[i][j].get_res_data())
        
        try:
            uncertainty_list = [self.wt_uncert, self.pv_uncert, np.zeros(self.nTimeslot)]
            for i in range(len(self.gen_list)):
                for j in range(len(self.gen_list[i])):
                    for step in range(self.nTimeslot):
                        self.total_max_power[step] += self.gen_list[i][j].max_power * (1+uncertainty_list[i][step])
                        self.total_min_power[step] += self.gen_list[i][j].min_power * (1-uncertainty_list[i][step])
                    
        except Exception as e:
            print("Error")
            print(e)
            
            print("Aggregator set_res_table method")
            print("Uncertainty does not exist")
            for i in range(len(self.gen_list)):
                for j in range(len(self.gen_list[i])):
                    for step in range(self.nTimeslot):
                        self.total_max_power[step] += self.gen_list[i][j].max_power 
                        self.total_min_power[step] += self.gen_list[i][j].min_power           
        try:
            if self.ess_list:
                self.res_table = pd.DataFrame(self.data_list,columns=['name', 'type', 'number', 'min_power', 'max_power',
                                                                 'capacity'])
            else:
                self.res_table = pd.DataFrame(self.data_list,columns=['name', 'type', 'number', 'min_power', 'max_power'])
        except:
            print("generate res table error")
    def get_res_table(self):
        
        try:
            return self.res_table
        except:
            
            self.set_res_table()
            return self.res_table
            
    def set_profile(self,N,count): 
        self.nScen = N
        for i in range(self.nRES):
            self.res_list[i].set_profile(N,count)
        
        self.set_res_cov()
            
    def set_res_cov(self):
        self.all_res_profile_xi = np.zeros([self.nTimeslot, self.nRES, self.nScen])
        self.sum_res_profile_xi = np.zeros([self.nTimeslot, self.nScen])
        self.sum_res_profile_sig = np.zeros([self.nTimeslot,1])
        self.sum_res_profile_sigi = np.zeros([self.nTimeslot,1])
        
        self.all_res_cov = []
        for t in range(self.nTimeslot):
            for i in range(self.nRES):
                for j in range(self.nScen):
                    self.all_res_profile_xi[t,i,j] = self.res_list[i].profile_xi[t,j]
                    self.sum_res_profile_xi[t,j] += self.res_list[i].profile_xi[t,j]
        for t in range(self.nTimeslot):
            self.all_res_cov.append(np.cov(self.all_res_profile_xi[t,:,:]))
            
        self.sum_res_profile_sig = self.sum_res_profile_xi.std(axis = 1)**2
        self.sum_res_profile_sigi = self.sum_res_profile_sig**(-0.5)
        
            
            
        print("Set_Res_Cov")
        
    
class WT:
    def __init__(self, name,code, count):
        self.name = f'WT{count}_{name}'
        self.type = 'WT'
        self.cvpp_name = name
        self.cvpp_code = code
        self.busNumber = count
        self.min_power = 0
        self.max_power = 0
        

    def set_power(self, max_power):
        # Unit [kW]
        self.max_power = max_power      
        
    def get_res_data(self):
        self.res_data = [
            self.name,
            self.type,
            self.busNumber,
            self.min_power,
            self.max_power
        ]  
        return self.res_data 
    
    def set_wt_profile(self, case_dict, model_dict):
        
        self.model_dict = model_dict
        self.case_dict = case_dict
        
        path = model_dict['path']
        self.nTimeslot = model_dict['nTimeslot']
        
        self.divide_factor = self.case_dict['divide_factor']
        
        n_total_scen = self.case_dict['n_total_scen']
        N_max = self.case_dict['N_max']
        OOS_max = self.case_dict['OOS_max']
        IR_max = self.case_dict['IR_max']
        
        wt_profile_dict = pd.read_excel(f"{path}/src/Data_Generation/발전실적(~2022.10.31)_modified.xlsx", 0, header=0)
        wt_data = wt_profile_dict.iloc[:,1:1 + self.nTimeslot].values
        wt_profile = wt_data / np.max(wt_data)
        wt_profile = wt_profile.astype(float)
        wff = np.maximum(wt_profile, 1e-6)
        
        # wt_profile_dict = io.loadmat(f'{path}/src/Data_Generation/AV_AEMO')
        # wff = wt_profile_dict['AV_AEMO2'][:, :self.nTimeslot]
        
        # Cutting off very extreme values

        cut_off_eps = 1e-2
        wff[wff<cut_off_eps] = cut_off_eps
        wff[wff>(1-cut_off_eps)] = 1 - cut_off_eps
        self.wff = wff
        
        '''
        # Logit-normal transformation (Eq. (1) in ref. [31])
        yy = np.log(wff/(1-wff))

        # Calculation of mean and variance, note that we increase the mean to have
        # higher wind penetration in our test-case
        
        mu = yy.mean(axis=0)
        # mu = yy.mean(axis=0) + 1.5 # Increase 1.5 for higher wind penetration
        cov_m = np.cov(yy, rowvar = False)
        std_yy = yy.std(axis=0).reshape(1, yy.shape[1])
        std_yy_T = std_yy.T
        sigma_m = cov_m / (std_yy_T @ std_yy)

        # Inverse of logit-normal transformation (Eq. (2) in ref. [31]) 
        R = np.linalg.cholesky(sigma_m).T
        '''
        # self.wt_rand_pattern = np.random.rand(self.nTimeslot, n_total_scen)/2\
        
        if self.case_dict['data_scenario'] == 'MC':
            self.wff_100 = self.wff[-100:,:]
            
            self.daily_total = np.sum(self.wff_100, axis=1)
    
            # 24시간 발전량 합의 백분위수 계산
            percentile_25 = np.percentile(self.daily_total, 25)
            percentile_75 = np.percentile(self.daily_total, 75)
            
            # 25~75% 범위에 해당하는 데이터 추출
            selected_data_indices = np.where((self.daily_total >= percentile_25) & (self.daily_total <= percentile_75))[0]
            self.selected_data = self.wff_100[selected_data_indices,:]
            
            nSelect = self.selected_data.shape[0]
                
            self.wt_rand_pattern = np.zeros((self.nTimeslot, n_total_scen))
            for t in range(self.nTimeslot):
                self.hist_wt = scipy.stats.rv_histogram(np.histogram(self.selected_data[:,t],bins= nSelect,density=True),density=True)
                self.wt_rand_pattern[t,:] = self.hist_wt.rvs(size=n_total_scen, random_state=0)
            self.wt_rand_pattern = self.wt_rand_pattern/np.max(self.wt_rand_pattern)
            
        elif self.case_dict['data_scenario'] == 'random':
            
            lb = - self.case_dict['wt_bound']
            ub = self.case_dict['wt_bound']
            self.base_profile = self.wff[-1, :]
            
            self.wt_rand_pattern = np.zeros((self.nTimeslot,n_total_scen))
            
            for t in range(self.nTimeslot):
                
                if self.case_dict['rand'] == 'uniform':
                    random_values = np.random.uniform(lb, ub, size=n_total_scen)
                elif self.case_dict['rand'] == 'gaussian':
                    random_values = self.case_dict['wt_bound']*np.random.randn(n_total_scen)
                
                self.wt_rand_pattern[t,:] = self.base_profile[t] * (1+random_values)
        
        
        
        # y = np.tile(mu, (n_total_scen,1)) + wt_rand_pattern @ R
        # Wind = (1 + np.exp(-y))**-1

        # Checking correlation, mean and true mean of data
        # corr_check_coeff = np.corrcoef(Wind, rowvar = False)
        # mu_Wind = Wind.mean(axis=0)
        # true_mean_Wind = (1+ np.exp(-mu))**-1

        # Reshaping the data structure
        if not self.case_dict['data_scenario'] == 'jeju':
            nWind = self.wt_rand_pattern
            self.nWind = nWind.reshape(self.nTimeslot, N_max + OOS_max, IR_max)
            
            # peak N and N' samples
            j = 0
            self.WPf_max = self.nWind[:,0:N_max,:]
            self.WPr_max = self.nWind[:,N_max:N_max + OOS_max, :]
        
        
        
    def set_profile(self, N, count):

        if self.case_dict['data_scenario'] == 'jeju':
            self.wff = self.wff * self.max_power
            self.wff = self.wff.transpose()
            OOS_max = self.case_dict['OOS_max']
            self.profile = self.wff[:,-N:]
            self.oos_profile = self.wff[:,-N-OOS_max:-N]
            print("Data for only historical data")
            
            #random_value = np.random.uniform(lb, ub, size= self.)
            
            #self.wt_rand_pattern = 
            
        else:
            self.WP = self.nWind[:,:,count]
            self.WPf = self.WPf_max[:,0:N,count]
            self.WPr = self.WPr_max[:,:,count]         
            
            self.WP = self.WP * self.max_power
            self.WPf = self.WPf * self.max_power
            self.WPr = self.WPr * self.max_power
            print("Data Generated by historical data")            
            self.profile = self.WPf[:,0:N]
               
        self.profile_mu = self.profile.mean(axis = 1)
        self.profile_mu = self.profile_mu.reshape(len(self.profile_mu),1)
        self.profile_xi = self.profile - np.tile(self.profile_mu,(1, self.profile.shape[1]))
        self.profile_xi = self.profile_xi / self.divide_factor
        self.profile_std = self.profile_xi.std(axis = 1)
        self.profile_std = self.profile_std.reshape(len(self.profile_std),1)
        self.profile_std = self.profile_std / self.divide_factor**2 
        
        if self.case_dict['data_scenario'] == 'jeju':
            self.oos_max_profile_xi = self.oos_profile - np.tile(self.profile_mu, (1, self.oos_profile.shape[1]))
            self.all_profile_xi = self.wff - np.tile(self.profile_mu, (1,self.wff.shape[1]))
        else:
            self.oos_max_profile_xi = self.WPr - np.tile(self.profile_mu, (1, self.WPr.shape[1]))
            self.all_profile_xi = self.WP - np.tile(self.profile_mu, (1,self.WP.shape[1]))
            
        self.oos_max_profile_xi = self.oos_max_profile_xi / self.divide_factor
        self.all_profile_xi = self.all_profile_xi / self.divide_factor
        
        self.worst_profile_xi = np.zeros(self.nTimeslot)
        self.worst_profile_std = np.zeros(self.nTimeslot)
        
        for i in range(self.nTimeslot):
            self.worst_profile_xi[i] = np.min(self.all_profile_xi[i,:])
            self.worst_profile_std[i] = self.all_profile_xi[i,:].std() 
        
        
        
        
class PV:
    def __init__(self, name,code, count):
        self.name = f'PV{count}_{name}'
        self.type = 'PV'
        self.cvpp_name = name
        self.cvpp_code = code
        self.busNumber = count
        self.min_power = 0
        self.max_power = 0
        
    def set_power(self, max_power):
        # Unit [kW]
        self.max_power = max_power
        
    def get_res_data(self):
        self.res_data = [
            self.name,
            self.type,
            self.busNumber,
            self.min_power,
            self.max_power
        ]  
        return self.res_data 
    
    def set_pv_profile(self, case_dict, model_dict):
        
        self.model_dict = model_dict
        self.case_dict = case_dict
        
        path = model_dict['path']
        self.nTimeslot = model_dict['nTimeslot']
        
        self.divide_factor = self.case_dict['divide_factor']
        
        n_total_scen = self.case_dict['n_total_scen']
        N_max = self.case_dict['N_max']
        OOS_max = self.case_dict['OOS_max']
        IR_max = self.case_dict['IR_max']
        
        wt_profile_dict = pd.read_excel(f"{path}/src/Data_Generation/발전실적(~2022.10.31)_modified.xlsx", 1, header=0)
        wt_data = wt_profile_dict.iloc[:,1:1 + self.nTimeslot].values
        wt_profile = wt_data / np.max(wt_data)
        wt_profile = wt_profile.astype(float)
        wff = np.maximum(wt_profile, 1e-6)
        
        # wt_profile_dict = io.loadmat(f'{path}/src/Data_Generation/AV_AEMO')
        # wff = wt_profile_dict['AV_AEMO2'][:, :self.nTimeslot]
        
        # Cutting off very extreme values

        cut_off_eps = 1e-2
        wff[wff<cut_off_eps] = cut_off_eps
        wff[wff>(1-cut_off_eps)] = 1 - cut_off_eps
        self.wff = wff
            
        '''
        # Logit-normal transformation (Eq. (1) in ref. [31])
        yy = np.log(wff/(1-wff))

        # Calculation of mean and variance, note that we increase the mean to have
        # higher wind penetration in our test-case
        
        mu = yy.mean(axis=0)
        # mu = yy.mean(axis=0) + 1.5 # Increase 1.5 for higher wind penetration
        cov_m = np.cov(yy, rowvar = False)
        std_yy = yy.std(axis=0).reshape(1, yy.shape[1])
        std_yy_T = std_yy.T
        sigma_m = cov_m / (std_yy_T @ std_yy)

        # Inverse of logit-normal transformation (Eq. (2) in ref. [31]) 
        R = np.linalg.cholesky(sigma_m).T
        '''
        # self.wt_rand_pattern = np.random.rand(self.nTimeslot, n_total_scen)/2\
        
        if self.case_dict['data_scenario'] == 'MC':
            self.wff_100 = self.wff[-100:,:]
            
            self.daily_total = np.sum(self.wff_100, axis=1)
    
            # 24시간 발전량 합의 백분위수 계산
            percentile_25 = np.percentile(self.daily_total, 25)
            percentile_75 = np.percentile(self.daily_total, 75)
            
            # 25~75% 범위에 해당하는 데이터 추출
            selected_data_indices = np.where((self.daily_total >= percentile_25) & (self.daily_total <= percentile_75))[0]
            self.selected_data = self.wff_100[selected_data_indices,:]
            
            nSelect = self.selected_data.shape[0]
                
            self.wt_rand_pattern = np.zeros((self.nTimeslot, n_total_scen))
            for t in range(self.nTimeslot):
                self.hist_wt = scipy.stats.rv_histogram(np.histogram(self.selected_data[:,t],bins= nSelect,density=True),density=True)
                self.wt_rand_pattern[t,:] = self.hist_wt.rvs(size=n_total_scen, random_state=0)
            self.wt_rand_pattern = self.wt_rand_pattern/np.max(self.wt_rand_pattern)
            
        elif self.case_dict['data_scenario'] == 'random':
            
            lb = - self.case_dict['pv_bound']
            ub = self.case_dict['pv_bound']
            self.base_profile = self.wff[-1,:]
            self.wt_rand_pattern = np.zeros((self.nTimeslot,n_total_scen))
            
            for t in range(self.nTimeslot):
                if self.case_dict['rand'] == 'uniform':
                    random_values = np.random.uniform(lb, ub, size=n_total_scen)
                elif self.case_dict['rand'] == 'gaussian':
                    random_values = self.case_dict['pv_bound']*np.random.randn(n_total_scen)
                
                self.wt_rand_pattern[t,:] = self.base_profile[t] * (1+random_values)
        
        
        # y = np.tile(mu, (n_total_scen,1)) + wt_rand_pattern @ R
        # Wind = (1 + np.exp(-y))**-1

        # Checking correlation, mean and true mean of data
        # corr_check_coeff = np.corrcoef(Wind, rowvar = False)
        # mu_Wind = Wind.mean(axis=0)
        # true_mean_Wind = (1+ np.exp(-mu))**-1

        # Reshaping the data structure
        if not self.case_dict['data_scenario'] == 'jeju':
            nWind = self.wt_rand_pattern
            self.nWind = nWind.reshape(self.nTimeslot, N_max + OOS_max, IR_max)
            
            # peak N and N' samples
            j = 0
            self.WPf_max = self.nWind[:,0:N_max,:]
            self.WPr_max = self.nWind[:,N_max:N_max + OOS_max, :]
        
    def set_profile(self, N, count):


        if self.case_dict['data_scenario'] == 'jeju':
            self.wff = self.wff * self.max_power
            self.wff = self.wff.transpose()
            OOS_max = self.case_dict['OOS_max']
            self.profile = self.wff[:,-N:]
            self.oos_profile = self.wff[:,-N-OOS_max:-N]
            
        else:
            self.WP = self.nWind[:,:,count]
            self.WPf = self.WPf_max[:,0:N,count]
            self.WPr = self.WPr_max[:,:,count]         
            
            self.WP = self.WP * self.max_power
            self.WPf = self.WPf * self.max_power
            self.WPr = self.WPr * self.max_power            
            self.profile = self.WPf[:,0:N]
               
        self.profile_mu = self.profile.mean(axis = 1)
        self.profile_mu = self.profile_mu.reshape(len(self.profile_mu),1)
        self.profile_xi = self.profile - np.tile(self.profile_mu,(1, self.profile.shape[1]))
        self.profile_xi = self.profile_xi / self.divide_factor
        self.profile_std = self.profile.std(axis = 1)
        self.profile_std = self.profile_std.reshape(len(self.profile_std),1)  / self.divide_factor**2 
        
        
        if self.case_dict['data_scenario'] == 'jeju':
            self.oos_max_profile_xi = self.oos_profile - np.tile(self.profile_mu, (1, self.oos_profile.shape[1]))
            self.all_profile_xi = self.wff - np.tile(self.profile_mu, (1,self.wff.shape[1]))
        else:
            self.oos_max_profile_xi = self.WPr - np.tile(self.profile_mu, (1, self.WPr.shape[1]))
            self.all_profile_xi = self.WP - np.tile(self.profile_mu, (1,self.WP.shape[1]))
        self.worst_profile_xi = np.zeros(self.nTimeslot)
        self.worst_profile_std = np.zeros(self.nTimeslot)
        
        for i in range(self.nTimeslot):
            self.worst_profile_xi[i] = np.min(self.all_profile_xi[i,:])
            self.worst_profile_std[i] = self.all_profile_xi[i,:].std() 
        
class ESS:
    def __init__(self, name,code, count, ess):
        self.name = f'ESS{count}_{name}'
        self.type = 'ESS'
        self.cvpp_name = name
        self.cvpp_code = code
        self.busNumber = count
        self.min_power = 0
        self.max_power = 0
        self.initSOC = ess['initSOC']
        self.termSOC = ess['termSOC']
        self.minSOC = ess['minSOC']
        self.maxSOC = ess['maxSOC']
        self.max_capacity = 0
        self.efficiency = ess['efficiency']

    def set_power(self, max_power):
        # Unit [kW]
        self.min_power = - max_power
        self.max_power = max_power
        
    def set_capacity(self, capacity):
        # Unit [kWh]
        self.max_capacity = capacity
                
    def get_res_data(self):
        self.res_data = [
            self.name,
            self.type,
            self.busNumber,
            self.min_power,
            self.max_power,
            self.max_capacity
        ]  
        return self.res_data 
    
class DG:
    def __init__(self, name,code, count, gen):
        self.name = f'DG{count}_{name}'
        self.type = 'DG'
        self.cvpp_name = name
        self.cvpp_code = code
        self.busNumber = count 
        
        self.min_power = gen['min_power']
        self.max_power = gen['max_power']
        self.a = gen['상수']
        self.b = gen['1차 계수']
        self.c = gen['2차 계수']
        
        self.ramp_up_limit = gen['ramp_up_limit'] 
        self.ramp_down_limit = gen['ramp_down_limit']
        self.max_capcity = 0

        self.start_up_cost = gen['start_up_cost']
        self.shut_down_cost = gen['shut_down_cost']
        self.fuel_cost = gen['fuel_cost']
            
        self.N_PIECE = gen['N_PIECE']
        
        self.set_slope()
        
        
    def set_slope(self):
             
        slopes_dg_cost = np.zeros(self.N_PIECE)
        
        self.max_power_per_piece = self.max_power / self.N_PIECE
        
        for i in range(self.N_PIECE):
            slopes_dg_cost[i] = (self.a + self.b * (self.max_power_per_piece * (i + 1))
                                       + self.c * (self.max_power_per_piece * (i + 1)) ** 2
                                       - self.a- self.b * self.max_power_per_piece * i
                                       - self.c * (self.max_power_per_piece * i) ** 2) / self.max_power_per_piece
        self.slopes = slopes_dg_cost
                    
    def get_res_data(self):
        self.res_data = [
            self.name,
            self.type,
            self.busNumber,
            self.min_power,
            self.max_power,
        ]  
        return self.res_data     
    
    
    

    
    
    
    