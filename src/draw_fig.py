# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:18:02 2023

@author: junhyeok
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from matplotlib import font_manager, rc
import matplotlib.ticker as ticker
import time
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def setPlotStyle():
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/Arial.ttf").get_name()
# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/NANUMBARUNPENR.TTF").get_name()
    mat.rcParams['font.family'] = "Times New Roman"
    mat.rcParams['font.size'] = 15
    mat.rcParams['legend.fontsize'] = 13
    mat.rcParams['legend.labelspacing'] = 0.2
    mat.rcParams['lines.linewidth'] = 2
    mat.rcParams['lines.color'] = 'r'
    mat.rcParams['axes.grid'] = 1     
    mat.rcParams['axes.xmargin'] = 0.1     
    mat.rcParams['axes.ymargin'] = 0.1     
    mat.rcParams["mathtext.fontset"] = "dejavuserif" #"cm", "stix", etc.
    mat.rcParams['figure.dpi'] = 500
    mat.rcParams['savefig.dpi'] = 500
    mat.rcParams['axes.unicode_minus'] = False


def save_fig(name, path):
    address = f'{path}/fig/'
    try:
        if not os.path.exists(address):
            os.makedirs(address)
    except OSError:
        print("Error: Creating directory. " + address)
        
    plt.savefig(address + time.strftime("%m%d%H%M") + f'_{name}.png', dpi=500, bbox_inches='tight')

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
    
class Single_Case_Plot:
    
    def __init__(self, vpp, opt_bid, model_dict, case_dict, path):
        
        self.vpp = vpp
        self.opt_bid = opt_bid
        
        self.model_dict = model_dict
        
        self.da_smp = opt_bid.dayahead_smp
        self.wt_list = opt_bid.wt_list
        self.pv_list = opt_bid.pv_list
        self.ess_list = opt_bid.ess_list
        self.dg_list = opt_bid.dg_list
        self.res_list = opt_bid.res_list
        
        self.wt_real = np.zeros([opt_bid.nWT, opt_bid.nTimeslot])
        self.pv_real = np.zeros([opt_bid.nPV, opt_bid.nTimeslot])
        
        self.nScen = opt_bid.nScen
        
        self.case_dict = case_dict
        self.is_case1 = self.case_dict['case'] == 1
        self.is_case2 = self.case_dict['case'] == 2
        self.is_case3 = self.case_dict['case'] == 3
        self.is_case4 = self.case_dict['case'] == 4
        
        self.is_bid_DRO = opt_bid.is_bid_DRO
        self.is_DRO_gamma = opt_bid.is_DRO_gamma
        self.is_bid_DRCC = opt_bid.is_bid_DRCC
        self.is_dg_reserve = opt_bid.is_dg_reserve
        self.is_ess_reserve = opt_bid.is_ess_reserve
        
        
        self.path = path
        
        self.nVPP = self.model_dict['nVPP']
        self.UNIT_TIME = self.case_dict['UNIT_TIME']
        self.nTimeslot = int (24 / self.UNIT_TIME)
        
        
        self.nDG = vpp[0].nDG
        self.nWT = vpp[0].nWT
        self.nPV = vpp[0].nPV
        self.nESS = vpp[0].nESS     
        self.nRES = vpp[0].nRES
        
        # Set the parameters for plot
        self.color_dict = {'da_smp': 'black',
                           'DG': 'red',
                           'WT': 'blue',
                           'PV': 'pink',
                           'ESS': 'orange',
                           'ESS_dis': 'orange',
                           'ESS_chg': 'purple',
                           'res': 'green'}
        
        self.legend_fontsize = 12 #8
        
        setPlotStyle()
        print("setPlotStyle()")
        
        
    def make_plot(self, P_dict, slack_dict, save_flag = False):
        
        self.P_bidSol = P_dict['bid']
        
        if self.is_bid_DRO:
            lambda_objSol = slack_dict['lambda_obj']
            s_objSol = slack_dict['s_obj']
            theta = self.opt_bid.theta
            worst_bid = np.zeros(self.nTimeslot)
            for j in range(self.nTimeslot):
                worst_bid[j] = theta[j]*lambda_objSol[j] / self.da_smp[j]
                for i in range(self.nScen):
                    worst_bid[j] += 1/self.nScen / self.da_smp[j] * s_objSol[i,j]            
            self.worst_bid = worst_bid
        if self.ess_list:
            self.P_essDisSol = P_dict['essDis']
            self.P_essChgSol = P_dict['essChg']
        
            if self.is_ess_reserve:
                self.RU_essDisSol = P_dict['RU_essdis']
                self.RD_essDisSol = P_dict['RD_essdis']
                self.RU_essChgSol = P_dict['RU_esschg']
                self.RD_essChgSol = P_dict['RD_esschg']
        
        if self.dg_list:
            self.P_dgSol = P_dict['dg']
            self.sum_P_dgSol = P_dict['sum_dg']
            self.RU_dgSol = P_dict['dg_ru']
            
        if len(self.vpp)==1:
            self.P_wtSol = np.zeros([self.nTimeslot])
            self.P_pvSol = np.zeros([self.nTimeslot])
            self.P_resSol = np.zeros([self.nTimeslot])
            self.P_essSol = np.zeros([self.nTimeslot])
             
            for i in range(self.nWT):
                self.P_wtSol = self.P_wtSol + self.wt_list[i].profile_mu.flatten()
            for i in range(self.nPV):    
                self.P_pvSol = self.P_pvSol + self.pv_list[i].profile_mu.flatten()
            self.P_resSol = self.P_wtSol + self.P_pvSol
                
            for i in range(self.nESS):
                self.P_essSol = self.P_essSol + self.P_essDisSol[i] - self.P_essChgSol[i]
        else:
            print("need to develop for more than vpp 1")
            
        plt.rcParams["figure.figsize"] = (8, 12)
        
        plt.figure()
        
        # Bid Graph
        x_tick = np.arange(self.nTimeslot) + 1
        ax1 = plt.subplot(211)
        #plt.title(r'$P_h^{DA}$ (kWh) & SMP [Won/kWh] ')
        ax1.plot(x_tick,self.P_bidSol, label=r'$\hat{P}_h^{bid}$', linestyle = 'dashed', color='blue', alpha = 0.8)
        if self.is_bid_DRO:
            ax1.plot(x_tick, self.P_bidSol - worst_bid, label=r'$P_h^{bid}$', color='blue', alpha = 0.8)
        
        self.plt_setting(r'$P_h^{bid}$ [kWh]')
        ax1.set_ylim([min(self.P_bidSol)*0.9,max(self.P_bidSol)*1.5])
        plt.legend(loc='upper left', ncol=3)
        ax2 = ax1.twinx()
        ax2.plot(x_tick,self.da_smp, label=r'$\pi_h^{DA}$', color='red', alpha = 0.8)
        ax2.set_ylim([min(self.da_smp)*0.3,max(self.da_smp)*1.1])
        
        plt.legend(loc='upper right', ncol=3)
        plt.grid(alpha=0.4, linestyle='--')
        plt.ylabel(r'$\pi_h^{DA} [Won/kWh]$')
        plt.xlim([1 -0.7, 1 + self.nTimeslot - 0.3])
        plt.xticks(np.arange(1, self.nTimeslot+1, 1 / self.UNIT_TIME)) #, fontsize=8)
        
        plt.subplot(212)
        
        #plt.title('The Active Power of Generators for Bid (kWh)')
        sum_bottom = np.zeros(self.nTimeslot)
        if len(self.vpp) == 1:
            plt.plot(x_tick,self.P_bidSol, label=r'$\hat{P}_h^{bid}$', linestyle='dashed', color='blue', alpha = 0.7)
            
            if self.opt_bid.ess_list:
                plt.bar(x_tick, -sum(self.P_essChgSol), label = r'$P_{s,h}^{chg}$', color ='purple', alpha = 0.7)
                plt.bar(x_tick, sum(self.P_essDisSol), label = r'$P_{s,h}^{dis}$', color ='orange', alpha = 0.7)
                
                sum_bottom = sum(self.P_essDisSol)
                plt.bar(x_tick, self.P_resSol - sum(self.P_essChgSol), bottom= sum_bottom, label = r'$\hat{P}_{v,h} + \hat{P}_{w,h} $', color ='green', alpha = 0.7)
                sum_bottom = sum_bottom + self.P_resSol - sum(self.P_essChgSol)
            else:
                plt.bar(x_tick, self.P_resSol, bottom= sum_bottom, label = r'$\hat{P}_{v,h} + \hat{P}_{w,h} $', color ='green', alpha = 0.7)
                sum_bottom = sum_bottom + self.P_resSol
            if self.dg_list:
                plt.bar(x_tick, sum(self.sum_P_dgSol),bottom= sum_bottom, label = r'$\hat{P}_{i,h}$', color =self.color_dict['DG'], alpha = 0.7)
                sum_bottom = sum_bottom + sum(self.sum_P_dgSol)
            plt.ylim([-max(sum(self.P_essChgSol))*1.1, max(sum_bottom)*1.4])

            self.plt_setting('Active Power  [kWh]')
        else:
            print(" Need to Develop for more than 2 VPP")
        
        # plt.rcParams["figure.figsize"] = (6,4.5)
        # plt.figure()
        # if self.case_dict['case'] == 2:
        #     if len(self.vpp) == 1:
                
        #         self.res_real = sum(self.wt_real) + sum(self.pv_real)
        #         self.res_upper = sum(self.wt_real * (1+self.wt_uncer)) + sum(self.pv_real * (1+self.pv_uncer)) -sum(self.P_essChgSol) + sum(self.P_essDisSol)
        #         self.res_under = sum(self.wt_real * (1-self.wt_uncer)) + sum(self.pv_real * (1-self.pv_uncer)) -sum(self.P_essChgSol) + sum(self.P_essDisSol)
                
        #         x = np.arange(self.nTimeslot)
                
        #         plt.fill(np.concatenate([x,x[::-1]]), 
        #                 np.concatenate([self.res_upper, self.res_under[::-1]]), alpha=.5, fc='green', ec='None', label = 'w uncertainty')
                
        #         plt.plot(np.arange(self.nTimeslot),self.P_bidSol, label='P_Bid', color='blue', alpha = 0.7)
                
        #         if self.opt_bid.ess_list:
        #             plt.bar(np.arange(self.nTimeslot), -sum(self.P_essChgSol), label = 'ESS_Chg', color ='purple', alpha = 0.7)
        #             plt.bar(np.arange(self.nTimeslot), sum(self.P_essDisSol), label = 'ESS_Dis', color ='orange', alpha = 0.7)
            
        #         plt.bar(np.arange(self.nTimeslot), self.P_resSol - sum(self.P_essChgSol), bottom= sum(self.P_essDisSol), label = 'WT + PV', color ='green', alpha = 0.7)
                
                    
        #         self.plt_setting('Power [kWh]')
        
        if save_flag:
            name = "active_power"
            save_fig(name, self.path)
            print(f"save the fig at location : {self.path}")
        
    def make_reserve_plot(self, P_dict, save_flag = False):
             
        plt.rcParams["figure.figsize"] = (8, 12)
        ax1 = plt.subplot(211)
        #plt.title('The Reserve of Generators for Uncertainty')
        
        profile_xi = np.zeros((self.nTimeslot, self.nScen))
        
        for res in range(self.nRES):
            profile_xi -= self.res_list[res].profile_xi
        profile_xi_max = np.max(profile_xi, axis = 1)
        self.profile_xi = profile_xi
        
        x_tick = np.arange(self.nTimeslot) + 1
        # ax1.plot(x_tick ,profile_xi_max - 0.1*self.P_bidSol, label=r'$\xi_r - P_{h}^{bid}/10$', color='green', alpha = 0.6, linewidth = 1.0)
        # ax1.plt.plot(np.arange(self.nTimeslot),profile_xi_max, label='RES_u', color='green', alpha = 0.7)
        
        sum_bottom = np.zeros(self.nTimeslot)
        if self.dg_list:
            if self.is_dg_reserve:
                ax1.bar(x_tick , sum(self.RU_dgSol), label = r'$R_{i,h}$', color =self.color_dict['DG'], alpha = 0.6)
                sum_bottom = sum_bottom + sum(self.RU_dgSol)
        
        if self.ess_list:
            if self.is_ess_reserve:
                ax1.bar(x_tick , sum(self.RU_essDisSol), bottom = sum_bottom, label = r'$R_{s,h}^{dis}$', color = self.color_dict['ESS_dis'], alpha = 0.6)
                sum_bottom = sum_bottom + sum(self.RU_essDisSol)
                ax1.bar(x_tick , sum(self.RU_essChgSol), bottom = sum_bottom, label = r'$R_{s,h}^{chg}$', color = self.color_dict['ESS_chg'], alpha = 0.6)
                sum_bottom = sum_bottom + sum(self.RU_essChgSol)
        self.plt_setting('Reserve Power [kWh]')
        
        ax1.set_ylim([0, max([max(profile_xi_max - 0.1*self.P_bidSol),max(sum_bottom)])*1.3])
        ax1.legend(loc='lower right', ncol=3) #, fontsize=8)
        ax2 = ax1.twinx()
        
        lhs, rhs, check_array, ratio = self.opt_bid.check_drcc_constraint()
        reliability_array = np.zeros(self.nTimeslot)
        for i in range(self.nTimeslot):
            reliability_array[i] = sum(check_array[i,:])
            
        ax2.plot(x_tick ,reliability_array, 'bo--', label=r'$\rho$', 
                 markersize=7, alpha = 0.8, linewidth = 1.0)
        ax2.set_ylim([70.0,104.0])
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(base=5.0))
        self.plt_setting('Prop. of safey conditions [%]')        

        
        plt.subplot(212)
        
        # plt.title('The Reserve of Generators Via Uncertainty')
        
        sum_bottom = np.zeros(self.nTimeslot)
        plt.bar(x_tick, 0.1*self.P_bidSol, bottom = sum_bottom, label= r'$P_h^{bid}/10$', color = 'blue', alpha = 0.6)
        sum_bottom = sum_bottom + 0.1*self.P_bidSol
        print("Pbid_bar")
        print(sum_bottom)
        if self.dg_list:
            if self.is_dg_reserve:
                plt.bar(x_tick, sum(self.RU_dgSol), bottom = sum_bottom, label = r'$R_{i,h}$', color =self.color_dict['DG'], alpha = 0.6)
                print("Pdg_ru")
                sum_bottom = sum_bottom + sum(self.RU_dgSol)
        
        if self.ess_list:
            if self.is_ess_reserve:
                plt.bar(x_tick, sum(self.RU_essDisSol), bottom = sum_bottom, label = r'$R_{s,h}^{dis}$', color = self.color_dict['ESS_dis'], alpha = 0.6)
                sum_bottom = sum_bottom + sum(self.RU_essDisSol)
                plt.bar(x_tick, sum(self.RU_essChgSol), bottom = sum_bottom, label = r'$R_{s,h}^{chg}$', color = self.color_dict['ESS_chg'], alpha = 0.6)
                sum_bottom = sum_bottom + sum(self.RU_essChgSol)
                
        # plt.bar(x_tick, 0.1*self.P_bidSol, bottom = sum_bottom, label= r'$P_h^{bid}/10$', color = 'blue', alpha = 0.4)
        
        plt.boxplot(profile_xi.transpose(), vert=True, patch_artist=True, widths = 0.5, medianprops = dict(color='k', alpha = 0.7),
                    boxprops=dict(facecolor = 'green', color = 'k', alpha = .7))
        plt.bar(x_tick, np.zeros(self.nTimeslot), color='green', alpha = 0.7, label=r'$-\hat{\xi}_h$')

        # Add the custom legend entry to the legend
        plt.legend(loc='best', ncol=3)

        plt.ylim([0, np.max(profile_xi_max)*1.3])
        
        plt.grid(alpha=0.4, linestyle='--')
        plt.ylabel('Box plot of uncertainty [kWh]')
        plt.xlim([1 -0.7, 1 + self.nTimeslot - 0.3])
        plt.xticks(np.arange(1, self.nTimeslot+1, 1 / self.UNIT_TIME)) #, fontsize=8)
        
        # self.plt_setting('Box plot of uncertainty [kWh]')
        if save_flag:
            name = "Reserve Plot"
            save_fig(name, self.path)
            print(f"save the fig at location : {self.path}")
            
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
        
        
    
    def make_3d_plot(self,data, label_name, save_flag = False, fig_size = (8,6)):    
        
        x_label = label_name[0]
        y_label = label_name[1]
        z_label = label_name[2]
        title = label_name[3]
        file_name = label_name[4]
        
        df = pd.DataFrame(data)
        
        # Initialize a 3D plot
        plt.rcParams["figure.figsize"] = fig_size
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate arrays for the dataframe values
        X = np.arange(df.shape[1])*0.01 + 0.01
        Y = np.arange(df.shape[0])*0.01 + 0.01
        X, Y = np.meshgrid(X, Y)
        Z = df.values
        
        # Plotting the 3D surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        
        # Add color bar which maps values to colors
        fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
        
        # Set labels
        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_label, labelpad=10)
        ax.set_zlabel(z_label, labelpad=5)
        # ax.set_title(title)
        
        # Show the plot            
        if save_flag:
            name = file_name
            save_fig(name, self.path)
            print(f"save the fig at location : {self.path}")
            
        plt.show()
    
    def make_igdt_plot(self,igdt_list, save_flag = False, fig_size = (18,6)):
        
        
        alpha = igdt_list[0]
        beta = igdt_list[1]
        obj = igdt_list[2]
        solution_time = igdt_list[3]
        
        
        plt.rcParams["figure.figsize"] = fig_size
        
        plt.figure()
        
        # Bid Graph
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        
        ax1 = plt.subplot(131)
        ax1.scatter(beta,alpha, marker = '*', s = 70, color='red', alpha = 0.8)
        ax1.set_xlabel(r"$\beta_r$")
        ax1.set_ylabel(r"$\alpha_{robust}$")
        ax1.grid(alpha=0.4, linestyle = '--')
        
        ax2 = plt.subplot(132)
        ax2.scatter(alpha,obj, marker ='*', s = 70, color='blue', alpha = 0.8)
        ax2.set_xlabel(r"$\alpha_{robust}$")
        ax2.set_ylabel(r"$f(x,\xi,\alpha_{robust})$")     
        ax2.grid(alpha=0.4, linestyle = '--')

        ax3 = plt.subplot(133)
        ax3.scatter(beta,solution_time, marker ='*', s = 70, color='green', alpha = 0.8)
        ax3.set_xlabel(r"$\beta_r$")
        ax3.set_ylabel("Solution Time [s]") 
        ax3.grid(alpha=0.4, linestyle = '--')

        if save_flag:
            name = "igdt_alpha_beta"
            save_fig(name, self.path)
            print(f"save the fig at location : {self.path}")

    def make_igdt_gridpiece_plot(self,igdt_list, save_flag = False, fig_size = (18,6)):
        
        
        alpha = igdt_list[0]
        grid_piece = igdt_list[1]
        obj = igdt_list[2]
        solution_time = igdt_list[3]
        
        
        plt.rcParams["figure.figsize"] = fig_size
        
        plt.figure()
        
        # Bid Graph
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        
        ax1 = plt.subplot(131)
        ax1.scatter(grid_piece,alpha, marker = '*', s = 70, color='red', alpha = 0.8)
        ax1.set_xlabel(r"$N_{piece}$")
        ax1.set_ylabel(r"$\alpha_{robust}$")
        ax1.set_ylim([min(alpha)*0.8, max(alpha)*1.3])
        ax1.grid(alpha=0.4, linestyle = '--')
        
        ax2 = plt.subplot(132)
        ax2.scatter(grid_piece,obj, marker ='*', s = 70, color='blue', alpha = 0.8)
        ax2.set_xlabel(r"$N_{piece}$")
        ax2.set_ylabel(r"$f(x,\xi,\alpha_{robust})$")      
        ax2.set_ylim([min(obj)*0.99, max(obj)*1.01])
        ax2.grid(alpha=0.4, linestyle = '--')
        
        ax3 = plt.subplot(133)
        ax3.scatter(grid_piece,solution_time, marker ='*', s = 70, color='green', alpha = 0.8)
        ax3.set_xlabel(r"$N_{piece}$")
        ax3.set_ylabel("Solution Time [s]") 
        ax3.grid(alpha=0.4, linestyle = '--')

        if save_flag:
            name = "igdt_grid_piece"
            save_fig(name, self.path)
            print(f"save the fig at location : {self.path}")

    
    def plt_setting(self,name):

        plt.legend(loc='best', ncol=3)
        plt.grid(alpha=0.4, linestyle='--')
        plt.ylabel(name)
        plt.xlim([1 -0.7, 1 + self.nTimeslot - 0.3])
        plt.xticks(np.arange(1, self.nTimeslot+1, 1 / self.UNIT_TIME)) #, fontsize=8)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        