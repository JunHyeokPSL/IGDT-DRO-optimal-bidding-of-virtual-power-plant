# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:18:02 2023

@author: junhyeok
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

def save_fig(name, path):
    address = f'{path}/fig/'
    try:
        if not os.path.exists(address):
            os.makedirs(address)
    except OSError:
        print("Error: Creating directory. " + address)
        
    plt.savefig(address + time.strftime("%m%d%H%M") + f'_{name}.png', dpi=300, bbox_inches='tight')

def draw_ax_vline(nTimeslot, max_vertical):
    for j in range(24, nTimeslot, 24):
        plt.axvline(j,0, max_vertical * 1.1, color='grey', linewidth = 1)
        
def draw_bar(nTimeslot, x_value, bottom_value, label_name, color):
    
    isEmpty = len(x_value) == 0
    if not isEmpty:
        if x_value.ndim >= 2:
            sum_x = sum(x_value)
        else:
            sum_x = x_value
        plt.bar(np.arange(nTimeslot), sum_x, 
                bottom = bottom_value,
                label= label_name, color = color)
            

def draw_bar_iteration(nTimeslot, x_bar, color_dict, cascading_flag):
    
    bottom = np.zeros(nTimeslot)
    
    res_list = []
    if cascading_flag:
        for i in range(len(x_bar)):
            res_list = res_list + list(x_bar[i].keys())
        res_list = list(dict.fromkeys(res_list))
        n_bar = len(res_list)
        
        x_dict = {}
        x_dict = {letter: 0 for letter in res_list}
        
        for i in range(len(x_bar)):
            for key_j in x_bar[i]:
                x_dict[key_j] = x_dict[key_j] + x_bar[i][key_j]   
                
    else:
        x_dict = x_bar
        
    i = 0
    for key in x_dict:
        draw_bar(nTimeslot, x_dict[key], bottom, key, color_dict[key])
        
        if x_dict[key].ndim >= 2:
            bottom = bottom + sum(x_dict[key])
        else:
            bottom = bottom + x_dict[key]
        i+=1
        
    plt.ylim([0, max(max(bottom)*1.1,0.01)])
    draw_ax_vline(nTimeslot, max(max(bottom)*1.1,0))
    
class Opt_Bid_Plot:
    
    def __init__(self, vpp, opt_bid, model_dict, case_dict, path):
        
        self.vpp = vpp
        self.opt_bid = opt_bid
        
        self.model_dict = model_dict
        self.da_smp = self.opt_bid.dayahead_smp
        
        self.case_dict = case_dict
        self.is_case1 = self.case_dict['case'] == 1
        self.is_case2 = self.case_dict['case'] == 2
        self.is_case3 = self.case_dict['case'] == 3
        self.is_case4 = self.case_dict['case'] == 4
        

        if self.is_case1:
            self.is_res_var = True
            self.is_uncertainty = False
        elif self.is_case2:
            self.is_res_var = True
            self.is_uncertainty = True
        elif self.is_case3:
            self.is_res_var = False
            self.is_uncertainty = False
        elif self.is_case4:
            self.is_res_var = True
            self.is_uncertainty = False           
        else:
            raise Exception("No Considered Case at init is_res_var")
        
        
        
        
        self.path = path
        
        self.nVPP = self.model_dict['nVPP']
        self.UNIT_TIME = self.case_dict['UNIT_TIME']
        self.nTimeslot = int (24 / self.UNIT_TIME)
        
        
        self.wt_list = opt_bid.wt_list
        self.pv_list = opt_bid.pv_list
        
        self.wt_real = np.zeros([opt_bid.nWT, opt_bid.nTimeslot])
        self.pv_real = np.zeros([opt_bid.nPV, opt_bid.nTimeslot])
        
        for j in range(opt_bid.nTimeslot):    
            for i in range(opt_bid.nWT):
                self.wt_real[i,j] = opt_bid.wt_list[i].max_power * opt_bid.wt_list[i].profile[j]
            for i in range(opt_bid.nPV):
                self.pv_real[i,j] = opt_bid.pv_list[i].max_power * opt_bid.pv_list[i].profile[j]
        
        try:
            self.uncertainty_dict = model_dict['uncertainty']
            self.wt_uncer = self.uncertainty_dict['wt']
            self.pv_uncer = self.uncertainty_dict['pv']
        except:
            print("No Uncertainty Sets in this case - Opt_Bid_Plot_init")
            
        if len(self.vpp) == 1:
            self.nWT = vpp[0].nWT
            self.nPV = vpp[0].nPV
            self.nESS = vpp[0].nESS       
        else:
            print("need to develop for more than vpp 1")

        # Set the parameters for plot
        self.color_dict = {'da_smp': 'red',
                           'WT': 'blue',
                           'PV': 'pink',
                           'ESS': 'orange',
                           'res': 'green'}
        
        self.legend_fontsize = 8
    def make_plot(self, P_dict):
        
        self.P_bidSol = P_dict['bid']
        
        if self.opt_bid.ess_list:
            self.P_essDisSol = P_dict['essDis']
            self.P_essChgSol = P_dict['essChg']
        
        
        if self.is_res_var:
            self.P_wtSol = P_dict['wt']
            self.P_pvSol = P_dict['pv']
        else:
            self.P_wtSol = self.opt_bid.P_wt
            self.P_pvSol = self.opt_bid.P_pv
        
        if len(self.vpp)==1:
            self.P_resSol = np.zeros([self.nTimeslot])
            self.P_essSol = np.zeros([self.nTimeslot])
             
            for i in range(self.nWT):
                self.P_resSol = self.P_resSol + self.P_wtSol[i] 
            for i in range(self.nPV):    
                self.P_resSol = self.P_resSol + self.P_pvSol[i]
                
            for i in range(self.nESS):
                self.P_essSol = self.P_essSol + self.P_essDisSol[i] - self.P_essChgSol[i]
        else:
            print("need to develop for more than vpp 1")
            
        plt.rcParams["figure.figsize"] = (6, 9)
        
        plt.figure()
        
        # Bid Graph
        
        ax1 = plt.subplot(211)
        plt.title('SMP [Won/kWh] & Bid (kWh)')
        ax1.plot(np.arange(self.nTimeslot),self.P_bidSol, label='P_Bid', color='blue', alpha = 0.8)
        self.plt_setting('P_Bid [kWh]')
        plt.legend(loc='upper left', ncol=3, fontsize=self.legend_fontsize)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(self.nTimeslot),self.da_smp, label='SMP', color='red', alpha = 0.8)
        self.plt_setting('SMP [Won/kWh]')
        
        
        plt.subplot(212)
        if len(self.vpp) == 1:
            plt.plot(np.arange(self.nTimeslot),self.P_bidSol, label='P_Bid', color='blue', alpha = 0.7)
            
            if self.opt_bid.ess_list:
                plt.bar(np.arange(self.nTimeslot), -sum(self.P_essChgSol), label = 'ESS_Chg', color ='purple', alpha = 0.7)
                plt.bar(np.arange(self.nTimeslot), sum(self.P_essDisSol), label = 'ESS_Dis', color ='orange', alpha = 0.7)
        
                plt.bar(np.arange(self.nTimeslot), self.P_resSol - sum(self.P_essChgSol), bottom= sum(self.P_essDisSol), label = 'WT + PV', color ='green', alpha = 0.7)
            else:
                plt.bar(np.arange(self.nTimeslot), self.P_resSol, bottom= np.zeros(self.nTimeslot), label = 'WT + PV', color ='green', alpha = 0.7)
            self.plt_setting('Power [kWh]')
        else:
            print(" Need to Develop for more than 2 VPP")
        
        plt.rcParams["figure.figsize"] = (6,4.5)
        plt.figure()
        if self.case_dict['case'] == 2:
            if len(self.vpp) == 1:
                
                self.res_real = sum(self.wt_real) + sum(self.pv_real)
                self.res_upper = sum(self.wt_real * (1+self.wt_uncer)) + sum(self.pv_real * (1+self.pv_uncer)) -sum(self.P_essChgSol) + sum(self.P_essDisSol)
                self.res_under = sum(self.wt_real * (1-self.wt_uncer)) + sum(self.pv_real * (1-self.pv_uncer)) -sum(self.P_essChgSol) + sum(self.P_essDisSol)
                
                x = np.arange(self.nTimeslot)
                
                plt.fill(np.concatenate([x,x[::-1]]), 
                        np.concatenate([self.res_upper, self.res_under[::-1]]), alpha=.5, fc='green', ec='None', label = 'w uncertainty')
                
                plt.plot(np.arange(self.nTimeslot),self.P_bidSol, label='P_Bid', color='blue', alpha = 0.7)
                
                if self.opt_bid.ess_list:
                    plt.bar(np.arange(self.nTimeslot), -sum(self.P_essChgSol), label = 'ESS_Chg', color ='purple', alpha = 0.7)
                    plt.bar(np.arange(self.nTimeslot), sum(self.P_essDisSol), label = 'ESS_Dis', color ='orange', alpha = 0.7)
            
                plt.bar(np.arange(self.nTimeslot), self.P_resSol - sum(self.P_essChgSol), bottom= sum(self.P_essDisSol), label = 'WT + PV', color ='green', alpha = 0.7)
                
                    
                self.plt_setting('Power [kWh]')
            
        
        
    def make_uncertainty_plot(self, P_dict):
        
        self.P_bidSol = P_dict['bid']
        
        if self.opt_bid.ess_list:
            self.P_essDisSol = P_dict['essDis']
            self.P_essChgSol = P_dict['essChg']
        
        
        if self.is_res_var:
            self.P_wtSol = P_dict['wt']
            self.P_pvSol = P_dict['pv']
        else:
            self.P_wtSol = self.opt_bid.P_wt
            self.P_pvSol = self.opt_bid.P_pv
        
        if len(self.vpp)==1:
            self.P_resSol = np.zeros([self.nTimeslot])
            self.P_essSol = np.zeros([self.nTimeslot])
             
            for i in range(self.nWT):
                self.P_resSol = self.P_resSol + self.P_wtSol[i] 
            for i in range(self.nPV):    
                self.P_resSol = self.P_resSol + self.P_pvSol[i]
                
            for i in range(self.nESS):
                self.P_essSol = self.P_essSol + self.P_essDisSol[i] - self.P_essChgSol[i]
        else:
            print("need to develop for more than vpp 1")
            
        plt.rcParams["figure.figsize"] = (6, 9)
        plt.figure()
        
        # Bid Graph
        
        ax1 = plt.subplot(211)
        plt.title('SMP [Won/kWh] & Bid (kWh)')
        ax1.plot(np.arange(self.nTimeslot),self.P_bidSol, label='P_Bid', color='blue', alpha = 0.8)
        self.plt_setting('P_Bid [kWh]')
        ax2 = ax1.twinx()
        ax2.plot(np.arange(self.nTimeslot),self.da_smp, label='SMP', color='red', alpha = 0.8)
        self.plt_setting('SMP [Won/kWh]')
        
        
        plt.subplot(212)
        if len(self.vpp) == 1:
            plt.plot(np.arange(self.nTimeslot),self.P_bidSol, label='P_Bid', color='blue', alpha = 0.7)
            
            if self.opt_bid.ess_list:
                plt.bar(np.arange(self.nTimeslot), -sum(self.P_essChgSol), label = 'ESS_Chg', color ='purple', alpha = 0.7)
                plt.bar(np.arange(self.nTimeslot), sum(self.P_essDisSol), label = 'ESS_Dis', color ='orange', alpha = 0.7)
        
            plt.bar(np.arange(self.nTimeslot), self.P_resSol - sum(self.P_essChgSol), bottom= sum(self.P_essDisSol), label = 'WT + PV', color ='green', alpha = 0.7)
            
                
            self.plt_setting('Power [kWh]')
        else:
            print(" Need to Develop for more than 2 VPP")    
        
        
    
    def plt_setting(self,name):
        
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.legend(loc='best', ncol=3, fontsize=self.legend_fontsize)
        plt.grid(alpha=0.4, linestyle='--')
        plt.ylabel(name)
        plt.xlim([-0.7, self.nTimeslot - 0.3])
        plt.xticks(np.arange(0, self.nTimeslot, 1 / self.UNIT_TIME), fontsize=8)