# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:03:27 2024

Ref : Distributionally Robust Energy Management for Islanded Microgrids With Variable Moment Information: An MISOCP Approach

@author: HOME
"""

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

class gurobi_SAA:
    def __init__(self, NAME, vpp, model_dict, case_dict):
        
        self.m = gp.Model(name=NAME)
        self.vpp = vpp
        
        self.model_dict = model_dict
        
        
        self.wt_list = self.vpp.wt_list
        self.pv_list = self.vpp.pv_list
        self.ess_list = self.vpp.ess_list
        self.dg_list = self.vpp.dg_list
        
        try:
            self.res_list = self.vpp.wt_list + self.vpp.pv_list
        except: 
            print("Fail to generate the res_list")
        
        self.nWT = self.vpp.nWT
        self.nPV = self.vpp.nPV
        self.nESS = self.vpp.nESS        
        self.nRES = self.nWT + self.nPV
        self.nDG = vpp.nDG
        self.dayahead_smp = self.vpp.da_smp_profile
        self.nICC = self.nDG
        self.N_PIECE = vpp.N_PIECE
        
        self.case_dict = case_dict
        self.is_case1 = self.case_dict['case'] == 1
        self.is_case2 = self.case_dict['case'] == 2
        self.is_case3 = self.case_dict['case'] == 3
        self.is_case4 = self.case_dict['case'] == 4 # case for DRO and DRCC
        self.is_case5 = self.case_dict['case'] == 5 # case for ess reserve
        self.is_case6 = self.case_dict['case'] == 6 # case for ess reserve
        self.is_case7 = self.case_dict['case'] == 7
        
        self.is_case_momentDRO = self.case_dict['case'] == 'moment_DRO'
            
        self.is_bid_var = False
        self.is_bid_DRO = False
        self.is_DRO_gamma = False
        
        self.is_bid_DRCC = False
        self.is_dg_reserve = False
        self.is_ess_reserve = False
        
        self.is_reserve_cost = False
        self.is_igdt_risk_averse = False
        
        if self.is_case1:
            self.is_bid_var= True
            self.is_uncertainty = False
            
        elif self.is_case_momentDRO:
            self.is_bid_var = True
            self.is_uncertainty = False
            self.is_bid_momentDRO = True
            self.is_dg_reserve = True
            self.is_ess_reserve = True
            
        elif self.is_case3:
            self.is_bid_var = True
            self.is_uncertainty = False
            #self.delta = self.model_dict['delta']
            self.is_bid_DRO = True
            self.is_DRO_gamma = True
            
        elif self.is_case4:
            self.is_bid_var = True
            self.is_uncertainty = False
            #self.delta = self.model_dict['delta']       
            self.is_bid_DRO = True
            self.is_DRO_gamma = True
            self.is_bid_DRCC = True
            self.is_dg_reserve = True
            
        elif self.is_case5:
            self.is_bid_var = True
            self.is_uncertainty = False
            #self.delta = self.model_dict['delta']       
            self.is_bid_DRO = True
            self.is_DRO_gamma = True
            self.is_bid_DRCC = True
            self.is_dg_reserve = True
            self.is_ess_reserve = True
            
        elif self.is_case6:
            self.is_bid_var = True
            self.is_uncertainty = False
            #self.delta = self.model_dict['delta']       
            self.is_bid_DRO = True
            self.is_DRO_gamma = True
            self.is_bid_DRCC = True
            self.is_dg_reserve = True
            self.is_ess_reserve = True
            self.is_reserve_cost = True
            
            
        elif self.is_case7:
            self.is_bid_var = True
            self.is_bid_DRO = True
            self.is_DRO_gamma = True
            self.is_bid_DRCC = True
            self.is_dg_reserve = True
            self.is_ess_reserve = True
            self.is_reserve_cost = True
            self.is_igdt_risk_averse = True
                
        else:
            raise Exception("No Considered Case at init is_res_var")
    
        
        
        # self.is_case_risk_averse = self.case_dict['bid_type'] == 'risk_averse'
        # self.is_igdt_risk_averse = self.case_dict['bid_type'] == 'igdt_risk_averse'
        # self.is_igdt_risk_seeking = self.case_dict['bid_type'] == 'igdt_risk_seeking'
        
        
        self.UNIT_TIME = self.case_dict['UNIT_TIME'] 
        self.nTimeslot = int (24 / self.UNIT_TIME)
        self.nScen = self.case_dict['N']        
        
        self.beta = case_dict['beta']
        self.GRID_PIECE = case_dict['GRID_PIECE']
        self.alpha_max = case_dict['alpha_max']
        
        self.check_set = {}
        
    def add_Variables(self):
        
        vpp = self.vpp

        if self.is_bid_var: 
            self.Pbid = self.m.addVars(self.nTimeslot, vtype =GRB.CONTINUOUS,
                              lb = [vpp.total_min_power[i] for i in range(self.nTimeslot)],
                              ub= [vpp.total_max_power[i] for i in range(self.nTimeslot)],
                              name='Pbid')
            
        else:
            raise Exception("No Considered Case at add_Variables")
                
        if self.dg_list:
            
            self.P_dg = self.m.addVars(vpp.nDG, self.N_PIECE, self.nTimeslot, vtype =GRB.CONTINUOUS,
                                      lb=[[0 for _ in range(self.nTimeslot) for _ in range(self.N_PIECE)] for i in range(self.nDG)],
                                      ub=[[self.dg_list[i].max_power_per_piece for _ in range(self.nTimeslot)  for _ in range(self.N_PIECE)] for i in range(self.nDG)],
                                      name='P_dg'
                                      ) 
            self.U_dg = self.m.addVars(self.nDG, self.nTimeslot, vtype=GRB.BINARY, name='U_dg')
            self.SU_dg = self.m.addVars(self.nDG, self.nTimeslot, vtype=GRB.BINARY, name='SU_dg')
            self.SD_dg = self.m.addVars(self.nDG, self.nTimeslot, vtype=GRB.BINARY, name='SD_dg')
            
            if self.is_dg_reserve:
                self.RU_dg = self.m.addVars(vpp.nDG, self.nTimeslot, vtype = GRB.CONTINUOUS,
                                           lb = [[0 for _ in range(self.nTimeslot)] for i in range(self.nDG)],
                                           ub = [[self.dg_list[i].ramp_up_limit for _ in range(self.nTimeslot)] for i in range(self.nDG)],
                                           name='RU_dg'   
                                           )
            
            
        if self.is_case1:
            print("")
            print("case 1 add Variables")
            print("gurobi_MILP add Variables")
            print("No Uncertainty Sets in this case")
            print("")
        
        
        if self.is_bid_SAA:
        
            # DRCC Objective Formulation
            # dual variables of Distributionally Robust Optimization
            self.s_obj = self.m.addVars(self.nScen, self.nTimeslot, lb = -1000000, ub = 1000000, vtype = gp.GRB.CONTINUOUS, name='s_obj')
            # self.lambda_obj = self.m.addVars(self.nTimeslot, vtype=gp.GRB.CONTINUOUS, lb = 0.0, ub= 1000000.0, name='lambda_obj')
            self.theta = self.case_dict['theta']     
        
        if self.is_bid_momentDRO:
            
            self.theta_0 = self.m.addVars(self.nTimeslot, lb = -100000, vtype = gp.GRB.CONTINUOUS, name='theta_0')
            self.theta_1 = self.m.addVars(self.nTimeslot, lb = -100000, vtype = gp.GRB.CONTINUOUS, name='theta_1')
            self.theta_2 = self.m.addVars(self.nTimeslot, lb = 0, vtype = gp.GRB.CONTINUOUS, name='theta_2')
            
            self.lambda_o = self.m.addVars(self.nTimeslot, lb = 0, vtype = gp.GRB.CONTINUOUS, name='lambda_o')
            self.tau_1 = self.m.addVars(self.nTimeslot, lb = 0, vtype = gp.GRB.CONTINUOUS, name='tau_1')
            self.tau_2 = self.m.addVars(self.nTimeslot, lb = 0, vtype = gp.GRB.CONTINUOUS, name='tau_2')
            
        
        if self.is_DRO_gamma:
            # dual variables of Distributionally Robust Optimization
            self.gamma_obj = self.m.addVars(2*self.nRES , self.nScen, self.nTimeslot, lb = 0,ub = 10000.0, vtype = gp.GRB.CONTINUOUS, name='gamma_obj')
       
        if self.is_bid_DRCC: 
            # DRCC
            # dual variables of Distributionally Robust Optimization
            
            Y_lb = [[[-1 for _ in range(self.nTimeslot)] for res in self.res_list] for _ in range(self.nDG)]
            Y_ub = [[[1 for _ in range(self.nTimeslot)] for res in self.res_list] for _ in range(self.nDG)]
            # Y_lb = [[[-res.max_power for _ in range(self.nTimeslot)] for res in self.res_list] for _ in range(self.nDG)]
            # Y_ub = [[[res.max_power for _ in range(self.nTimeslot)] for res in self.res_list] for _ in range(self.nDG)]
            self.Y = self.m.addMVar((self.nDG, self.nRES, self.nTimeslot), vtype = gp.GRB.CONTINUOUS, 
                               lb = Y_lb,
                               ub = Y_ub,
                               name='Y')
        
        
        if self.ess_list:
            self.P_essChg = self.m.addVars(vpp.nESS, self.nTimeslot, vtype =GRB.CONTINUOUS,
                                     lb= 0,
                                     ub=[[-self.ess_list[i].min_power for _ in range(self.nTimeslot)] for i in range(self.nESS)],
                                     name='P_essChg'
                                     )
            self.P_essDis = self.m.addVars(vpp.nESS, self.nTimeslot, vtype =GRB.CONTINUOUS,
                                     lb= 0,
                                     ub=[[self.ess_list[i].max_power for _ in range(self.nTimeslot)] for i in range(self.nESS)],
                                     name='P_essDis'
                                     )
            self.U_essChg = self.m.addVars(vpp.nESS, self.nTimeslot, vtype =GRB.BINARY, name='U_essChg')
            self.U_essDis = self.m.addVars(vpp.nESS, self.nTimeslot, vtype =GRB.BINARY, name='U_essDis')
            
            if self.is_ess_reserve:
                self.RU_essDis = self.m.addVars(vpp.nESS, self.nTimeslot, 
                                                lb = 0, 
                                                ub = [[self.ess_list[i].max_power*2 for _ in range(self.nTimeslot)] for i in range(self.nESS)],
                                                vtype=GRB.CONTINUOUS, name = 'RU_essDis')     

                self.RD_essDis = self.m.addVars(vpp.nESS, self.nTimeslot, 
                                                lb = 0, 
                                                ub = [[self.ess_list[i].max_power for _ in range(self.nTimeslot)] for i in range(self.nESS)],
                                                vtype=GRB.CONTINUOUS, name = 'RD_essDis')
                self.RU_essChg = self.m.addVars(vpp.nESS, self.nTimeslot, 
                                                lb = 0, 
                                                ub = [[self.ess_list[i].max_power for _ in range(self.nTimeslot)] for i in range(self.nESS)],
                                                vtype=GRB.CONTINUOUS, name = 'RU_essChg')
                self.RD_essChg = self.m.addVars(vpp.nESS, self.nTimeslot, 
                                                lb = 0, 
                                                ub = [[self.ess_list[i].max_power*2 for _ in range(self.nTimeslot)] for i in range(self.nESS)],
                                                vtype=GRB.CONTINUOUS, name = 'RD_essChg')
                
        
        if self.is_igdt_risk_averse:      
            self.alpha = self.m.addVar(lb = 0, ub = self.case_dict['alpha_max'] , vtype = GRB.CONTINUOUS, name= 'alpha')
            self.delta = self.m.addVars(self.GRID_PIECE, vtype = GRB.BINARY, name="delta")
            
            self.alpha_dx = self.m.addVars(self.GRID_PIECE, vtype = GRB.CONTINUOUS,
                                          lb = 0,
                                          name = 'alpha_dx')
            self.Pbid_dx = self.m.addVars(self.GRID_PIECE, self.nTimeslot, vtype = GRB.CONTINUOUS,
                                          lb = 0,
                                          name = 'Pbid_dx')
            self.z = self.m.addVars(self.nTimeslot, vtype = GRB.CONTINUOUS, lb = 0, name = 'z')
            self.dz = self.m.addVars(self.nTimeslot, vtype = GRB.CONTINUOUS, lb = 0, name = 'dz')
            
            
            
        else:
            print("Does not Cosidered alpha")              
    
    def add_bid_constraints(self):
        
        for j in range(self.nTimeslot):
            wt_sum = gp.quicksum(self.wt_list[i].profile_mu[j] for i in range(self.nWT))
            pv_sum = gp.quicksum(self.pv_list[i].profile_mu[j] for i in range(self.nPV))
            dg_sum = gp.quicksum(self.P_dg[i, k, j] for i in range(self.nDG) for k in range(self.N_PIECE))
            ess_dis_sum = gp.quicksum(self.P_essDis[i, j] for i in range(self.nESS))
            ess_chg_sum = gp.quicksum(self.P_essChg[i, j] for i in range(self.nESS))
            ess_sum = ess_dis_sum - ess_chg_sum
            # if self.is_ess_reserve:
            #     ess_dis_r_sum = gp.quicksum(self.RU_essDis[i, j] for i in range(self.nESS))
            #     ess_chg_r_sum = gp.quicksum(self.RD_essChg[i, j] for i in range(self.nESS))
            #     ess_sum = ess_sum + ess_dis_r_sum + ess_chg_r_sum
            
            self.m.addConstr(
                self.Pbid[j] == wt_sum + pv_sum + dg_sum + ess_sum,
                name=f'const_bid{j}'
            )   
            
        print("Add Bid Constraint")
                       
    def add_dg_constraints(self):
        

        for j in range(self.nTimeslot):
            
            for i in range(self.nDG):
                
                if self.is_dg_reserve:
                    # Min/Max limit
                    self.m.addConstr(gp.quicksum(self.P_dg[i, k, j] for k in range(self.N_PIECE))                                 
                                          >= self.U_dg[i, j] * self.dg_list[i].min_power, name=f'const_dg_min{j},{i}')
                    self.m.addConstr(gp.quicksum(self.P_dg[i, k, j] for k in range(self.N_PIECE))
                                     + self.RU_dg[i,j]
                                          <= self.U_dg[i, j] * self.dg_list[i].max_power,  name=f'const_dg_max{j},{i}')
                else:
                    # Min/Max limit
                    self.m.addConstr(gp.quicksum(self.P_dg[i, k, j] for k in range(self.N_PIECE))                                 
                                          >= self.U_dg[i, j] * self.dg_list[i].min_power, name=f'const_dg_min{j},{i}')
                    self.m.addConstr(gp.quicksum(self.P_dg[i, k, j] for k in range(self.N_PIECE))                                     
                                          <= self.U_dg[i, j] * self.dg_list[i].max_power,  name=f'const_dg_max{j},{i}')                    
                    
                    
                # Start Up / Shut Down Limit
                self.m.addConstr(
                    self.SU_dg[i, j] + self.SD_dg[i, j] <= 1, name=f'const_dg_su_sd_{i},{j}')
                
                if j == self.nTimeslot - 1:
                    continue
                self.m.addConstr(
                    self.SU_dg[i, j + 1] - self.SD_dg[i, j + 1] == self.U_dg[i, j + 1] - self.U_dg[i, j], name=f'const_dg_su_sd_u_{i},{j}')        
         
        # Ramp Limit   
        for j in range(self.nTimeslot - 1):
            # 발전기 ramp up down 제약(No reserve included yet)
            for i in range(self.nDG):
                
                if self.is_dg_reserve:
                    self.m.addConstr(gp.quicksum(self.P_dg[i, k, j + 1] for k in range(self.N_PIECE))
                                          - gp.quicksum(self.P_dg[i, k, j] for k in range(self.N_PIECE))
                                          + self.RU_dg[i,j]                                       
                                          <= self.dg_list[i].ramp_up_limit * self.UNIT_TIME, name=f'const_ramp_up[{j},{i}]' )
                    self.m.addConstr(gp.quicksum(self.P_dg[i, k, j + 1] for k in range(self.N_PIECE))                                 
                                          - gp.quicksum(self.P_dg[i, k, j] for k in range(self.N_PIECE))    
                                          >= -self.dg_list[i].ramp_down_limit * self.UNIT_TIME, name=f'const_ramp_down[{j},{i}]')
                else:
                    self.m.addConstr(gp.quicksum(self.P_dg[i, k, j + 1] for k in range(self.N_PIECE))
                                          - gp.quicksum(self.P_dg[i, k, j] for k in range(self.N_PIECE))                                                                                 
                                          <= self.dg_list[i].ramp_up_limit * self.UNIT_TIME, name=f'const_ramp_up[{j},{i}]' )
                    self.m.addConstr(gp.quicksum(self.P_dg[i, k, j + 1] for k in range(self.N_PIECE))                                 
                                          - gp.quicksum(self.P_dg[i, k, j] for k in range(self.N_PIECE))    
                                          >= -self.dg_list[i].ramp_down_limit * self.UNIT_TIME, name=f'const_ramp_down[{j},{i}]')           
    def add_ess_contraints(self):
        # ESS
        ess_list = self.ess_list
        if ess_list:
            for i in range(self.nESS):

                for j in range(self.nTimeslot):    
                    
                    # Ess minmax with preventing Discharging and Charging simultaneously
            
                    self.m.addConstr(self.P_essDis[i, j] <= self.U_essDis[i, j] * ess_list[i].max_power,
                                    name=f'const_ess{i}_{j}_power_max')
                    self.m.addConstr(self.P_essChg[i, j] <= self.U_essChg[i, j] * - ess_list[i].min_power,
                                    name=f'const_ess{i}_{j}_power_min')
                    self.m.addConstr(self.U_essDis[i, j] + self.U_essChg[i, j] <= 1,
                                    name=f'const_ess{i}_{j}_on/off')
                    
                    
                if self.is_ess_reserve:
                    for j in range(self.nTimeslot):    
                
                        self.m.addConstr(self.RU_essDis[i,j] <= self.ess_list[i].max_power - self.P_essDis[i, j],
                                        name=f'const_ess{i}_{j}_RUdis_max')
                                            
                        self.m.addConstr(self.RD_essDis[i,j] <=  self.P_essDis[i, j],
                                        name=f'const_ess{i}_{j}_RDdis_max')
                        
                        self.m.addConstr(self.RU_essChg[i,j] <=  self.P_essChg[i, j],
                                        name=f'const_ess{i}_{j}_RUchg_max')
                        
                        self.m.addConstr(self.RD_essChg[i,j] <= self.ess_list[i].max_power - self.P_essChg[i, j],
                                        name=f'const_ess{i}_{j}_RDchg_max')
                    
                        # SoC min max constraint
                        self.m.addConstr(ess_list[i].initSOC 
                                         - sum((self.P_essDis[i, k] + self.RU_essDis[i,k] )* self.UNIT_TIME 
                                               / ess_list[i].efficiency / ess_list[i].max_capacity
                                                    for k in range(j + 1)) 
                                         + sum((self.P_essChg[i, k] - self.RD_essChg[i,k] )* self.UNIT_TIME 
                                               * ess_list[i].efficiency / ess_list[i].max_capacity
                                                    for k in range(j + 1)) <= ess_list[i].maxSOC, 
                                              name=f'const_ess{i}_{j}_soc_max')
                        
                        self.m.addConstr(ess_list[i].initSOC
                                              - sum((self.P_essDis[i, k] + self.RU_essDis[i,k] ) * self.UNIT_TIME 
                                                    / ess_list[i].efficiency / ess_list[i].max_capacity
                                                    for k in range(j + 1))
                                              + sum((self.P_essChg[i, k] - self.RU_essChg[i,k] ) * self.UNIT_TIME 
                                                    * ess_list[i].efficiency / ess_list[i].max_capacity
                                                    for k in range(j + 1)) >= ess_list[i].minSOC,
                                              name=f'const_ess{i}_{j}_soc_min')
                    # SoC terminal constraint
                    self.m.addConstr(ess_list[i].initSOC
                                          - sum((self.P_essDis[i, k] + self.RU_essDis[i,k] ) * self.UNIT_TIME 
                                                / ess_list[i].efficiency / ess_list[i].max_capacity
                                                for k in range(self.nTimeslot))
                                          + sum((self.P_essChg[i, k] + self.RU_essChg[i,k] ) * self.UNIT_TIME 
                                                * ess_list[i].efficiency / ess_list[i].max_capacity
                                                for k in range(self.nTimeslot)) == ess_list[i].termSOC,
                                         name=f'const_ess{i}_term')
                else:                    
                    for j in range(self.nTimeslot):
                        # SoC min max constraint
                        self.m.addConstr(ess_list[i].initSOC 
                                         - sum(self.P_essDis[i, k]* self.UNIT_TIME 
                                               / ess_list[i].efficiency / ess_list[i].max_capacity
                                                    for k in range(j + 1)) 
                                         + sum(self.P_essChg[i, k] * self.UNIT_TIME 
                                               * ess_list[i].efficiency / ess_list[i].max_capacity
                                                    for k in range(j + 1)) <= ess_list[i].maxSOC, 
                                              name=f'const_ess{i}_{j}_soc_max')
                        
                        self.m.addConstr(ess_list[i].initSOC
                                              - sum(self.P_essDis[i, k] * self.UNIT_TIME 
                                                    / ess_list[i].efficiency / ess_list[i].max_capacity
                                                    for k in range(j + 1))
                                              + sum(self.P_essChg[i, k] * self.UNIT_TIME 
                                                    * ess_list[i].efficiency / ess_list[i].max_capacity
                                                    for k in range(j + 1)) >= ess_list[i].minSOC,
                                              name=f'const_ess{i}_{j}_soc_min')
                    # SoC terminal constraint
                    self.m.addConstr(ess_list[i].initSOC
                                          - sum(self.P_essDis[i, k]  * self.UNIT_TIME 
                                                / ess_list[i].efficiency / ess_list[i].max_capacity
                                                for k in range(self.nTimeslot))
                                          + sum(self.P_essChg[i, k] * self.UNIT_TIME 
                                                * ess_list[i].efficiency / ess_list[i].max_capacity
                                                for k in range(self.nTimeslot)) == ess_list[i].termSOC,
                                         name=f'const_ess{i}_term')          
    def set_SAA_obj_constraints(self):
        print("start set_saa_obj_constriants")
        # Now consider the only WT case
        if self.is_bid_SAA:
            for j in range(self.nTimeslot):
                # Constraints that c \xi <= si regardless gamma term with (h-H\xi)
                for n in range(self.nScen):
                    lhs = 0
                    for i in range(self.nRES):
                        
                        if not self.is_igdt_risk_averse:
                            lhs += self.dayahead_smp[j] * self.res_list[i].profile_xi[j][n]
                        else:
                            lhs += (1-self.alpha) * self.dayahead_smp[j] * self.res_list[i].profile_xi[j][n]
                    self.m.addConstr(self.s_obj[n,j] >= lhs, name = f'Const_si_dual_{j}_{n}')
                # Add Dual Norm Constraint for obj
                       
            print("end set_saa_obj_constriants")   
        else:
            raise Exception("No Considered Case at DRO obj")  
    
    def set_SAA_Constraint(self):       
       
       M = 1000000
       self.eps = self.case_dict['eps']       
       
       self.s_c = self.m.addMVar((self.nScen,self.nTimeslot), lb= 0, name = 's_c')
       self.t_c = self.m.addMVar(self.nTimeslot, lb= 0, name = 't_c')
       
       self.y_c = self.m.addMVar((self.nScen, self.nTimeslot), vtype = gp.GRB.BINARY, name='y_c' )
              
       
       print("start drjcc")
       
       for t in range(self.nTimeslot):
           
           lhs_x = 0.1 * self.Pbid[t]
           if self.is_dg_reserve:               
               lhs_x += gp.quicksum(self.RU_dg[gg,t] for gg in range(self.nDG))
               
           if self.is_ess_reserve:
               lhs_x += gp.quicksum(self.RU_essDis[ee,t] + self.RU_essChg[ee,t] for ee in range(self.nESS))
               
                    
           
           for n in range(self.nScen):
               # -1 for (b- Amat) & -1 for -Y\xi
               
               lhs_m = M * self.y_c[n,t]
               
               rhs = gp.quicksum( -1 * self.res_list[rr].profile_xi[t][n] for rr in range(self.nRES))
               
               self.m.addConstr(lhs_x + lhs_m >= rhs , name=f"max_const{t}_{n}")
           
           # Y에 대해 어떻게 반영 할지 고민필요함. 
           # Y 반영 시에는 주석 제거
           #for ww in range(self.nRES):
               
                
               # self.m.addConstr( gp.quicksum(self.Y[gg,ww,t] for gg in range(self.nDG)) ==  - 1, name = f'uncertainty_balance{ww}_{t}') 
               
               # self.m.addConstr(self.lhs_CC[ww,t] == gp.quicksum(-1 * -1 * self.Y[gg, ww,t] for gg in range(self.nDG)))
               
            # self.m.addGenConstrNorm( self.lhs_dual_CC[t], self.lhs_CC[:,t], GRB.INFINITY, f"Const_dualnorm_CC_{t}")
           
           # rhs_sc = self.theta[t] * self.nScen * self.lhs_dual_CC[t]
           rhs_sc = np.floor(self.eps * self.nScen).astype(int) 
           lhs_sc = gp.quicksum(self.y_c[nn,t] for nn in range(self.nScen))
           self.m.addConstr( lhs_sc <= rhs_sc, name = f"const_SAA_{t}")
           
           if t % 5 ==0 :
               print(f"iteration {t} of sum drjcc")
       print("finish max constraint of DRJCC")
    
    def set_DRCC_Constraint_JCC(self):
        
        
        self.eps = self.case_dict['eps']
        
        self.k_c = self.m.addMVar((self.nDG, self.nTimeslot), name = 'k_c')
        self.s_c = self.m.addMVar((self.nScen,self.nTimeslot), lb= 0, name = 's_c')
        self.t_c = self.m.addMVar(self.nTimeslot, lb= 0, name = 't_c')
        
        
        self.lhs_CC = self.m.addMVar((self.nDG, self.nRES, self.nTimeslot), lb = -1000000, ub = 1000000, vtype=gp.GRB.CONTINUOUS, name='lhs_CC')
        self.lhs_dual_CC = self.m.addMVar((self.nDG, self.nTimeslot), name='lhs_dual_CC')
        
        self.lhs_max = self.m.addMVar((self.nDG, self.nScen, self.nTimeslot), lb = 0, name='lhs_max')
        self.lhs_sum = self.m.addMVar((self.nDG, self.nScen, self.nTimeslot), lb = -10000000, name='lhs_sum')
        
        print("start drjcc")
    
        
        for t in range(self.nTimeslot):
            sum_sc = gp.quicksum(self.s_c[ss, t] for ss in range(self.nScen))
            lhs_sc = self.eps * self.nScen * self.t_c[t] - sum_sc
            
            for gg in range(self.nDG):
                lhs_x = self.RU_dg[gg,t] + self.k_c[gg,t]
                
                
                for n in range(self.nScen):                
                    
                    # -1 for (b- Amat) & -1 for -Y\xi
                    lhs_u = - gp.quicksum( self.Y[gg,rr,t] * self.res_list[rr].profile_xi[t][n] for rr in range(self.nRES))
                    
                    self.m.addConstr( self.lhs_sum[gg,n,t] == lhs_x + lhs_u, name= f"lhs_sum{gg}_{t}_{n}")
                    self.m.addGenConstrMax(self.lhs_max[gg,n,t], [self.lhs_sum[gg,n,t], 0.0], name=f"max_assign{gg}_{t}_{n}")
                    
                    rhs = self.t_c[t] - self.s_c[n,t]
                
                    self.m.addConstr(self.lhs_max[gg,n,t] >= rhs , name=f"max_const{gg}_{t}_{n}")
                
                for ww in range(self.nRES):                
                    
                    self.m.addConstr( gp.quicksum(self.Y[k,ww,t] for k in range(self.nDG)) ==  - 1, name = f'uncertainty_balance{ww}_{t}') 
                    
                    self.m.addConstr(self.lhs_CC[gg,ww,t] == -1 * -1 * self.Y[gg, ww,t] )
                self.m.addGenConstrNorm( self.lhs_dual_CC[gg,t], self.lhs_CC[gg,:,t], GRB.INFINITY, f"Const_dualnorm_CC_{t}_{n}")
                
                # rhs_sc = 0.0000005 * self.nScen * self.lhs_dual_CC[gg,t]
                rhs_sc = self.theta[t] * self.nScen * self.lhs_dual_CC[gg,t]
                
                #self.theta[t]/1000 -> self.theta[t]
                self.m.addConstr( lhs_sc >= rhs_sc, name = f"const_DRCC_{t}")
                
            if t % 5 ==0 :
                print(f"iteration {t} of sum drjcc")
        print("finish max constraint of DRJCC")
    
    def set_base_Objectives(self):
        print("start set_base_objectives")
        if self.is_bid_SAA:
            self.obj1 = gp.quicksum(self.dayahead_smp[j]*self.Pbid[j] for j in range(self.nTimeslot))
            # self.obj2 = gp.quicksum(self.theta[j] * self.lambda_obj[j] for j in range(self.nTimeslot))
            self.obj3 = gp.quicksum(1/self.nScen * self.s_obj[i,j] for i in range(self.nScen) for j in range(self.nTimeslot))
            
            self.obj_sum_without_bid = - self.obj3
            obj = self.obj1 - self.obj3
        
        else:
            obj = gp.quicksum(self.dayahead_smp[j]*self.Pbid[j] for j in range(self.nTimeslot))
                    #self.obj = gp.quicksum(self.dayahead_smp[j] for j in range(self.nTimeslot))
     
            print("No Considered Any Case at base_Objectives")  
        
        # Ramp Obj
        
        if self.dg_list:
            if self.is_dg_reserve:
                self.obj_dg_ramp = gp.quicksum(self.dayahead_smp[j] * self.RU_dg[gg,j] for gg in range(self.nDG) for j in range(self.nTimeslot))
                
                if not self.is_reserve_cost:
                    self.obj_sum_without_bid = self.obj_sum_without_bid + self.obj_dg_ramp
                    obj = obj + self.obj_dg_ramp
                    print("Obj - DG - RAMP")
                
        if self.ess_list:
            if self.is_ess_reserve:
                self.obj_ess_dis_ramp = gp.quicksum(self.RU_essDis[i, j] * self.dayahead_smp[j] for i in range(self.nESS) for j in range(self.nTimeslot))
                self.obj_ess_chg_ramp  = gp.quicksum(self.RU_essChg[i, j] * self.dayahead_smp[j] for i in range(self.nESS) for j in range(self.nTimeslot))
                self.obj_ess_ramp = self.obj_ess_dis_ramp + self.obj_ess_chg_ramp 
                
                if not self.is_reserve_cost:
                    obj = obj + self.obj_ess_ramp
                    self.obj_sum_without_bid = self.obj_sum_without_bid + self.obj_ess_ramp
                    print("Obj - ESS - RAMP")

        
        
        if self.dg_list:
            
            self.dg_gen_cost =  gp.quicksum(self.dg_list[i].slopes[k] * self.P_dg[i, k, j] * self.dg_list[i].fuel_cost
                                            for i in range(self.nDG) for k in range(self.N_PIECE) for j in range(self.nTimeslot))
            self.dg_start_shut_cost = gp.quicksum((self.U_dg[i, j] * self.dg_list[i].a * self.UNIT_TIME
                                       + self.SU_dg[i, j] * self.dg_list[i].start_up_cost
                                       + self.SD_dg[i, j] * self.dg_list[i].shut_down_cost) * self.dg_list[i].fuel_cost
                                       for i in range(self.nDG) for j in range(self.nTimeslot))
            
            if self.is_dg_reserve:
                self.dg_ramp_cost = gp.quicksum(self.dg_list[gg].b * self.RU_dg[gg,j] for gg in range(self.nDG) for j in range(self.nTimeslot))
                
                self.dg_obj_cost = self.dg_gen_cost + self.dg_start_shut_cost + self.dg_ramp_cost
            else: 
                self.dg_obj_cost = self.dg_gen_cost + self.dg_start_shut_cost
        
            obj = obj - self.dg_obj_cost
            try:
                self.obj_sum_without_bid = self.obj_sum_without_bid - self.dg_obj_cost
            except:
                print("No obj_sum_without_bid before this")
                self.obj_sum_without_bid = - self.dg_obj_cost
        print("end set_base_objectives")       
        return obj
    
    def set_Objectives(self):
        
        base_case_obj = self.set_base_Objectives() 
        if not self.is_igdt_risk_averse:
            self.obj = base_case_obj
        
        else:
            self.Fr = self.add_igdt_constraints()
            self.obj = self.alpha
        self.set_obj = self.m.setObjective(self.obj, GRB.MAXIMIZE)
        return self.obj
    
    def add_igdt_constraints(self):
        
        print("start add_igdt_constraints")
        
        if self.is_igdt_risk_averse:
            # Add the SOS1 constraint: Only one delta_n can be non-zero
            self.m.addSOS(GRB.SOS_TYPE1, self.delta.values())
            
            self.m.addConstr(gp.quicksum(self.delta[n] for n in range(self.GRID_PIECE)) == 1, name='SOS_const')
            delta_gap = self.alpha_max/self.GRID_PIECE
            self.alpha_piece = np.arange(self.GRID_PIECE + 1) * delta_gap # 0 to GRID_PIECE
            
            #SOS1 Constraint 
            # alpha Constraint (as omega in (A2)-(A4) of two-stage IGDT-stochastic model for optimal scheduling of energy communities with inteligent parking lots - appendix)
            for n in range(self.GRID_PIECE):
                self.m.addGenConstrIndicator(self.delta[n], True, self.alpha >= self.alpha_piece[n], name=f"delta_{n}_lower")
                self.m.addGenConstrIndicator(self.delta[n], True, self.alpha <= self.alpha_piece[n+1], name=f"delta_{n}_upper")
                self.m.addConstr(self.alpha_dx[n] <= delta_gap * self.delta[n], name='alpha_dx_const')
                
            self.m.addConstr(self.alpha == gp.quicksum( self.delta[n] * self.alpha_piece[n] + self.alpha_dx[n]   for n in range(self.GRID_PIECE)), name='alpha_piece_const')
            
            # Pbid const - y constraint
            for t in range(self.nTimeslot):
                Pbid_min = self.vpp.total_min_power[t]
                Pbid_max = self.vpp.total_max_power[t]
                
                self.m.addConstr(self.Pbid[t] == Pbid_min + gp.quicksum(self.Pbid_dx[n,t] for n in range(self.GRID_PIECE))
                                 , name=f"const_Pbid_piecewise_{t}")
                for n in range(self.GRID_PIECE):
                    self.m.addConstr(self.Pbid_dx[n,t] <= (Pbid_max - Pbid_min) * self.delta[n], name=f"const_Pbid_dx_{n}_{t}")
                    
            
                # Bi-liear term - dummy variable
                self.m.addConstr(self.z[t] == Pbid_min * self.alpha 
                                 + gp.quicksum(self.alpha_piece[n]* self.Pbid_dx[n,t] for n in range(self.GRID_PIECE)) 
                                 + self.dz[t], name=f"z_eq_const_{t}") 
                self.m.addConstr(self.dz[t] >= gp.quicksum( delta_gap * self.Pbid_dx[n,t] for n in range(self.GRID_PIECE))
                                 + (Pbid_max - Pbid_min)* gp.quicksum(self.alpha_dx[n] - delta_gap*self.delta[n] for n in range(self.GRID_PIECE))
                                 , name = f"z_geq_const_{t}")
                self.m.addConstr(self.dz[t] <= (Pbid_max - Pbid_min)* gp.quicksum(self.alpha_dx[n] for n in range(self.GRID_PIECE))
                                 , name = f"z_leq_const1_{t}")
                
                self.m.addConstr(self.dz[t] <= gp.quicksum(delta_gap * self.delta[n]  for n in range(self.GRID_PIECE))
                                 , name = f"z_leq_const2_{t}")
            
            
            self.profit = gp.quicksum(self.dayahead_smp[j]*self.Pbid[j] for j in range(self.nTimeslot))
            self.decline_profit = gp.quicksum(self.dayahead_smp[j]*self.z[j] for j in range(self.nTimeslot))
            obj = self.profit - self.decline_profit + self.obj_sum_without_bid             
            
            self.m.addConstr(obj >= (1-self.beta)*self.base_obj,
                             name = 'const_igdt_risk_averse_profit')
            
            print("add_igdt_risk_averse_constraints sucessfully") 
            
        print("End add_igdt_constarints")
        return obj
            
        # elif self.is_igdt_risk_seeking:
        #     self.revenue =gp.quicksum(self.da_smp[j]*self.Pbid[j] for j in range(self.nTimeslot))
        #     self.m.addConstr(self.revenue >= (1+self.delta)*self.base_obj,
        #                      name = 'const_igdt_risk_seeking_cost')
        
        #     print("add_igdt_risk_seeking_constraints sucessfully") 
        
    def solve(self, tol, timelimit=None):
        
        mip_gap = tol[0]
        feas_tol = tol[1]
        self.m.setParam(GRB.Param.MIPGap, mip_gap)
        self.m.setParam(GRB.Param.FeasibilityTol, feas_tol)
        
        
        sol = self.m.optimize()
        
        if self.m.status == GRB.OPTIMAL:
            print("Optimal Solution:")
        elif self.m.status == GRB.INFEASIBLE:
            print("Model is infeasible")
            
            self.m.computeIIS()
            if self.m.IISMinimal:
                print("Model is infeasible.")
                infeasible_constrs = self.m.getConstrs()
                infeasible_vars = self.m.getVars()
                for constr in infeasible_constrs:
                    if constr.IISConstr:
                        print(f"Infeasible constraint: {constr.ConstrName}")
                for var in infeasible_vars:
                    if var.IISLB > 0 or var.IISUB > 0:
                        print(f"Infeasible variable: {var.VarName}")
                        
        return sol, self.m.objVal
        
    def get_sol(self):
        
        P_BidSol = np.zeros([self.nTimeslot])
        
        P_essDisSol = np.zeros([self.nESS, self.nTimeslot])
        P_essChgSol = np.zeros([self.nESS, self.nTimeslot])
        U_essDisSol = np.zeros([self.nESS, self.nTimeslot])
        U_essChgSol = np.zeros([self.nESS, self.nTimeslot])
        RU_essDisSol = np.zeros([self.nESS, self.nTimeslot])
        RD_essDisSol = np.zeros([self.nESS, self.nTimeslot])
        RU_essChgSol = np.zeros([self.nESS, self.nTimeslot])
        RD_essChgSol = np.zeros([self.nESS, self.nTimeslot])
        
        
        P_dgSol = np.zeros([self.nDG, self.N_PIECE, self.nTimeslot])
        sum_P_dgSol = np.zeros([self.nDG, self.nTimeslot])
        U_dgSol = np.zeros([self.nDG, self.nTimeslot])
        SU_dgSol = np.zeros([self.nDG, self.nTimeslot])
        SD_dgSol = np.zeros([self.nDG, self.nTimeslot])
        RU_dgSol = np.zeros([self.nDG, self.nTimeslot])
        
        lambda_objSol = np.zeros([self.nTimeslot])
        s_objSol = np.zeros([self.nScen, self.nTimeslot])
        sum_s_objSol = np.zeros([self.nTimeslot])
        gamma_objSol = np.zeros([2*self.nRES, self.nScen, self.nTimeslot])
        
        s_obj_exists = "s_obj[0,0]" in [var.VarName for var in self.m.getVars()]
        lambda_obj_exists = "lambda_obj[0]" in [var.VarName for var in self.m.getVars()]
        gamma_obj_exists = "gamma_obj[0,0,0]" in [var.VarName for var in self.m.getVars()] 
        YY_Sol = np.zeros([self.nRES, self.nScen, self.nTimeslot])
        
        self.nICC = self.nDG
        s_cSol = np.zeros([self.nScen, self.nTimeslot])
        t_cSol = np.zeros([self.nTimeslot])
        y_cSol = np.zeros((self.nScen, self.nTimeslot))
        
        lhs_maxSol = np.zeros([self.nScen, self.nTimeslot])
        lhs_dual_CCSol = np.zeros([self.nTimeslot])
        Y_Sol = np.zeros([self.nDG, self.nRES, self.nTimeslot])
        
        Z_Sol = np.zeros([self.nTimeslot])
        dZ_Sol = np.zeros([self.nTimeslot])
        alpha_Sol = 0
        delta_Sol = np.zeros([self.GRID_PIECE])
        alpha_dxSol = np.zeros([self.GRID_PIECE])      
        s_c_exists = "s_c[0,0,0]" in [var.VarName for var in self.m.getVars()]
        t_c_exists = "t_c[0,0]" in [var.VarName for var in self.m.getVars()]
        lhs_dual_CC_exists = "lhs_dual_CC[0,0]" in [var.VarName for var in self.m.getVars()]
        lhs_max_exists = "lhs_max[0,0,0]" in [var.VarName for var in self.m.getVars()]
        
        P_dict = {}
        U_dict = {}
        slack_dict = {}
        
        
        for j in range(self.nTimeslot):
            
            try:
                P_BidSol[j] = self.m.getVarByName(f"Pbid[{j}]").X
                
            except Exception as e:
                print("Error")
                print(e)
                P_BidSol[j] = self.Pbid[j]
  
            try:
                if s_obj_exists:
                    for i in range(self.nScen):
                        s_objSol[i,j] = self.m.getVarByName(f"s_obj[{i},{j}]").X                        
                        sum_s_objSol[j] += s_objSol[i,j]
                else:
                    pass
                        
            except Exception as e:
                print("Error")
                print(e)
                print("s_obj error")
                print("s_obj error")
                for i in range(self.nScen):
                    s_objSol[i,j] = 0
            try:
                if lambda_obj_exists:    
                    lambda_objSol[j] = self.m.getVarByName(f"lambda_obj[{j}]").X
                else:
                    pass
                
            except Exception as e:
                print("Error")
                print(e)
                print("lambda_obj error")
                print("lambda_obj error")
                lambda_objSol[j] = 0
            
            try:
                if gamma_obj_exists:
                    for i in range(self.nRES*2):
                        for n in range(self.nScen):
                            gamma_objSol[i,n,j] = self.m.getVarByName(f"gamma_obj[{i},{n},{j}]").X
                else:
                    pass

            except Exception as e:
                print("Error")
                print(e)
                print("gamma_obj error")
                print("gamma_obj error")
                
            if self.is_bid_SAA:
                for n in range(self.nScen):
                    y_cSol[n,j] = self.m.getVarByName(f"y_c[{n},{j}]").X    
                    
            if self.is_DRO_gamma:
                for i in range(self.nRES):
                    for n in range(self.nScen):
                        YY_Sol[i,n,j] = self.m.getVarByName(f"YY_{i}_{n}_{j}").X
            
            for i in range(self.nESS):
                P_essDisSol[i,j] = self.m.getVarByName(f"P_essDis[{i},{j}]").X
                P_essChgSol[i,j] = self.m.getVarByName(f"P_essChg[{i},{j}]").X
                U_essDisSol[i,j] = self.m.getVarByName(f"U_essDis[{i},{j}]").X
                U_essChgSol[i,j] = self.m.getVarByName(f"U_essChg[{i},{j}]").X
                
                if self.is_ess_reserve:
                    RU_essDisSol[i,j] = self.m.getVarByName(f"RU_essDis[{i},{j}]").X
                    RD_essDisSol[i,j] = self.m.getVarByName(f"RD_essDis[{i},{j}]").X
                    RU_essChgSol[i,j] = self.m.getVarByName(f"RU_essChg[{i},{j}]").X
                    RD_essChgSol[i,j] = self.m.getVarByName(f"RD_essChg[{i},{j}]").X
                    
            for i in range(self.nDG):
                for k in range(self.N_PIECE):
                    P_dgSol[i,k,j] = self.m.getVarByName(f"P_dg[{i},{k},{j}]").X
                    sum_P_dgSol[i,j] += P_dgSol[i,k,j]
                U_dgSol[i,j] = self.m.getVarByName(f"U_dg[{i},{j}]").X 
                SU_dgSol[i,j] = self.m.getVarByName(f"SU_dg[{i},{j}]").X 
                SD_dgSol[i,j] = self.m.getVarByName(f"SD_dg[{i},{j}]").X
                
                if self.is_dg_reserve:
                    RU_dgSol[i,j] = self.m.getVarByName(f"RU_dg[{i},{j}]").X
            
            if self.is_bid_DRCC:
                
                t_cSol[j] = self.m.getVarByName(f"t_c[{j}]").X
                
                for gg in range(self.nDG):
                    lhs_dual_CCSol[j] = self.m.getVarByName(f"lhs_dual_CC[{j}]").X 
                    for n in range(self.nScen):
                        s_cSol[n,j] = self.m.getVarByName(f"s_c[{n},{j}]").X
                        lhs_maxSol[n,j] = self.m.getVarByName(f"lhs_max[{n},{j}]").X
                                            
                        
                for i in range(self.nDG):
                    for r in range(self.nRES):
                        Y_Sol[i,r,j] = self.m.getVarByName(f"Y[{i},{r},{j}]").X 
                        
            if self.is_igdt_risk_averse:
                Z_Sol[j] = self.m.getVarByName(f"z[{j}]").X
                dZ_Sol[j] = self.m.getVarByName(f"dz[{j}]").X
                alpha_Sol = self.m.getVarByName("alpha").X
                
        if self.is_igdt_risk_averse:
            for n in range(self.GRID_PIECE):                    
                alpha_dxSol[n] = self.m.getVarByName(f"alpha_dx[{n}]").X
                delta_Sol[n] = self.m.getVarByName(f"delta[{n}]").X
        
        P_dict['bid'] = P_BidSol
        
        if self.dg_list:
            P_dict['dg'] = P_dgSol
            P_dict['sum_dg'] = sum_P_dgSol
            U_dict['dg'] = U_dgSol
            U_dict['dg_su'] = SU_dgSol
            U_dict['dg_sd'] = SD_dgSol
            
            P_dict['dg_ru'] = RU_dgSol
            
            
        if self.ess_list:
            P_dict['essDis'] = P_essDisSol
            P_dict['essChg'] = P_essChgSol
            U_dict['essDis'] = U_essDisSol
            U_dict['essChg'] = U_essChgSol
            
            if self.is_ess_reserve:
                P_dict['RU_essdis'] = RU_essDisSol    
                P_dict['RD_essdis'] = RD_essDisSol
                P_dict['RU_esschg'] = RU_essChgSol
                P_dict['RD_esschg'] = RD_essChgSol
        if self.is_bid_DRO:
            slack_dict['lambda_obj'] = lambda_objSol
            slack_dict['s_obj'] = s_objSol 
        
        if self.is_DRO_gamma:
            slack_dict['gamma_obj'] = gamma_objSol
            slack_dict['YY'] = YY_Sol
            
        if self.is_bid_DRCC:
            slack_dict['s_c'] = s_cSol
            slack_dict['t_c'] = t_cSol 
            slack_dict['lhs_dual_CC'] = lhs_dual_CCSol
            slack_dict['lhs_max'] = lhs_maxSol
            slack_dict['Y'] = Y_Sol
        if self.is_igdt_risk_averse:
            slack_dict['alpha'] = alpha_Sol        
            slack_dict['z'] = Z_Sol
            slack_dict['dz'] = dZ_Sol
            slack_dict['alpha_dx'] = alpha_dxSol
            slack_dict['delta'] = delta_Sol
            
        if self.is_bid_SAA:
            slack_dict['s_obj'] = s_objSol
            
            slack_dict['y_c'] = y_cSol
            
              
        self.P_dict = P_dict
        self.U_dict = U_dict
        
        return P_dict, U_dict, slack_dict
    
    def optimize(self, mip_gap, feas_tol):
        
        self.add_Variables()
        self.add_bid_constraints()
        # self.add_smp_constraints()
        # self.add_res_constraints()
        
        self.add_dg_constraints()
        self.add_ess_contraints()
        # self.add_igdt_constraints()
        
        if self.is_bid_SAA:
            self.set_SAA_obj_constraints()
            self.set_SAA_Constraint()
        
        if self.is_bid_DRO:
            self.set_DRO_obj_constraints()
        if self.is_bid_DRCC:
            self.set_DRCC_Constraint()
            # self.set_DRCC_Constraint()
            
        obj_eq = self.set_Objectives()   
        if self.is_igdt_risk_averse:
            self.add_igdt_constraints()
        
        
        time_start_op = time.time()
        sol, obj = self.solve([mip_gap, feas_tol])
        time_end_op = time.time()
        self.opt_solve_time = time_end_op - time_start_op
        print("Optimization Duration Time:", self.opt_solve_time)
        
        
        P_dict, U_dict, slack_dict = self.get_sol()        
        
        obj_dict = {}
        
        
        obj1 = np.zeros(self.nTimeslot)
        obj2 = np.zeros(self.nTimeslot)
        obj3 = np.zeros(self.nTimeslot)
        obj3_full = np.zeros((self.nScen, self.nTimeslot))
        obj_dg_gen_cost = np.zeros((self.nDG, self.N_PIECE, self.nTimeslot))
        obj_dg_sum_gen_cost = np.zeros((self.nDG, self.nTimeslot))
        
        obj_dg_run_cost = np.zeros((self.nDG, self.nTimeslot))
        obj_dg_cost = np.zeros((self.nDG, self.nTimeslot))       
        obj_igdt_decline = np.zeros(self.nTimeslot)
        if self.is_case1:
            PbidSol = P_dict['bid']
            for j in range(self.nTimeslot):
                    obj1[j] = self.dayahead_smp[j] * PbidSol[j]
 
        else:    
            try:
                PbidSol = P_dict['bid']
                
                if self.is_bid_DRO:
                    lambda_objSol = slack_dict['lambda_obj']
                s_objSol = slack_dict['s_obj']
                
                RbidSol = np.zeros(self.nTimeslot)
                if self.is_dg_reserve:
                    RbidSol += sum(P_dict['dg_ru'])
                    
                if self.is_ess_reserve:
                    RbidSol += sum(P_dict['RU_essdis']) + sum(P_dict['RU_esschg']) 
                

                
                for j in range(self.nTimeslot):
                    if self.is_reserve_cost:
                        obj1[j] = self.dayahead_smp[j] * (PbidSol[j])
                    else:
                        obj1[j] = self.dayahead_smp[j] * (PbidSol[j] + RbidSol[j])
                    
                    
                    if self.is_bid_DRO:
                        obj2[j] = self.theta[j] * lambda_objSol[j]
                    
                    for i in range(self.nScen):
                        obj3_full[i,j] = 1/self.nScen * s_objSol[i,j]
                        obj3[j] += obj3_full[i,j]

            except Exception as e:
                print("Error")
                print(e)
                print("obj_dict generate failed")
        
        if self.dg_list: 
            for j in range(self.nTimeslot):
                for i in range(self.nDG):
                    for k in range(self.N_PIECE):
                        obj_dg_gen_cost[i,k,j] = self.dg_list[i].slopes[k] * P_dict['dg'][i,k,j] * self.dg_list[i].fuel_cost 
                        obj_dg_sum_gen_cost[i,j] += obj_dg_gen_cost[i,k,j]
                    obj_dg_run_cost[i,j] = U_dict['dg'][i,j] * self.dg_list[i].a * self.UNIT_TIME * self.dg_list[i].fuel_cost 
                    + U_dict['dg_su'][i,j] * self.dg_list[i].start_up_cost * self.dg_list[i].fuel_cost 
                    + U_dict['dg_sd'][i,j] * self.dg_list[i].shut_down_cost * self.dg_list[i].fuel_cost 
                    obj_dg_cost[i,j] = obj_dg_sum_gen_cost[i,j] + obj_dg_run_cost[i,j]
                    
                    if self.is_dg_reserve:
                        obj_dg_cost[i,j] = obj_dg_cost[i,j] + self.dg_list[i].b * self.P_dict['dg_ru'][i,j] 
        
        if self.is_igdt_risk_averse:
            for j in range(self.nTimeslot):
                obj_igdt_decline[j] = self.dayahead_smp[j] * slack_dict['z'][j]
                
            obj_dict['obj_risk_averse'] = obj_igdt_decline        
                
        obj_dict['obj1'] = obj1 
        obj_dict['obj2'] = obj2
        obj_dict['obj3'] = obj3
        obj_dict['obj3_full'] = obj3_full
        
        obj_dict['dg_sum'] = obj_dg_sum_gen_cost
        obj_dict['dg'] = obj_dg_gen_cost
        obj_dict['dg_susd'] = obj_dg_run_cost
        obj_dict['dg_cost'] = obj_dg_cost
        
        
        
        self.obj_dict = obj_dict
        self.P_dict = P_dict
        self.U_dict = U_dict
        self.slack_dict = slack_dict
        
        
        return sol, obj_dict, P_dict, U_dict, slack_dict
    

    def check_drcc_constraint(self):
        
        Pbid = self.P_dict['bid']
        Rdg_i = self.P_dict['dg_ru']
        Rdg = np.zeros(self.nTimeslot)
        
        Ress_dis_i = self.P_dict['RU_essdis']
        Ress_chg_i = self.P_dict['RU_esschg']
        
        Ress = np.zeros(self.nTimeslot)
                
        if self.is_dg_reserve:
            for i in range(self.nDG):
                Rdg += Rdg_i[i]
        
        if self.is_ess_reserve:
            for i in range(self.nESS):
                Ress += Ress_dis_i [i] + Ress_chg_i[i]
                
        profile_xi = np.zeros((self.nTimeslot, self.nScen))
        for i in range(self.nRES):
            profile_xi += self.res_list[i].profile_xi
            
        res_min_values = np.zeros(self.nTimeslot)
        res_max_values = np.zeros(self.nTimeslot)   
        for t in range(self.nTimeslot):
            res_min_values[t] = np.min(profile_xi[t,:])
            res_max_values[t] = np.max(profile_xi[t,:])

        lhs_array = np.zeros((self.nTimeslot, self.nScen))
        rhs_array = np.zeros(self.nTimeslot)
        check_array = np.zeros((self.nTimeslot, self.nScen))
        count = 0
        
        for t in range(self.nTimeslot):
            rhs_array[t] = 0.1*Pbid[t]
            if self.is_dg_reserve:
                rhs_array[t] += Rdg[t] #self.test_const # + Pbid[t]*0.1
            if self.is_ess_reserve:
                rhs_array[t] += Ress[t] #self.test_const # + Pbid[t]*0.1

            for n in range(self.nScen):
                lhs_array[t,n] = - profile_xi[t,n]
            
                if lhs_array[t,n] <= rhs_array[t] + 0.01:
                    check_array[t,n] = 1
                    count = count + 1
                else:
                    check_array[t,n] = 0
         
        ratio = count / (self.nTimeslot * self.nScen) 
        return lhs_array, rhs_array, check_array, ratio
    
    def add_smp_constraints(self):
        
        if self.is_case1 or self.is_case2 or self.is_case4:
            for j in range(self.nTimeslot):
                self.da_smp[j] = self.dayahead_smp[j]
            print("ADD smp constraints as fixed")
            
        elif self.is_case3:
            for j in range(self.nTimeslot):
                if self.is_igdt_risk_averse:
                    self.m.addConstr(self.da_smp[j] == (1-self.alpha) * self.dayahead_smp[j],
                                     name= f'const_da_smp{j}_uncertainty_igdt_risk_averse') 
                elif self.is_igdt_risk_seeking:
                    self.m.addConstr(self.da_smp[j] == (1+self.alpha) * self.dayahead_smp[j],
                                     name= f'const_da_smp{j}_uncertainty_igdt_risk_seeking')
        else: 
            raise Exception("No Considered Case at smp constraints Constraints")     
    
    def set_igdt_params(self, base_obj, beta):
        self.base_obj = base_obj
        self.beta = beta        
    
    def oos_test(self):
        
        self.nOOS = self.case_dict['OOS_sim']
        Pbid = self.P_dict['bid']
        Rdg_i = self.P_dict['dg_ru']
        Rdg = np.zeros(self.nTimeslot)
        
        Ress_dis_i = self.P_dict['RU_essdis']
        Ress_chg_i = self.P_dict['RU_esschg']
        
        Ress = np.zeros(self.nTimeslot)
                
        if self.is_dg_reserve:
            for i in range(self.nDG):
                Rdg += Rdg_i[i]
        
        if self.is_ess_reserve:
            for i in range(self.nESS):
                Ress += Ress_dis_i [i] + Ress_chg_i[i]
                
        profile_xi = np.zeros((self.nTimeslot, self.nOOS))
        for i in range(self.nRES):
            profile_xi += self.res_list[i].oos_profile_xi[:,:self.nOOS]
            
        # res_min_values = np.zeros(self.nTimeslot)
        # res_max_values = np.zeros(self.nTimeslot)   
        # for t in range(self.nTimeslot):
        #     res_min_values[t] = np.min(profile_xi[t,:])
        #     res_max_values[t] = np.max(profile_xi[t,:])

        lhs_array = np.zeros((self.nTimeslot, self.nOOS))
        rhs_array = np.zeros(self.nTimeslot)
        check_array = np.zeros((self.nTimeslot, self.nOOS))
        count = 0
        
        for t in range(self.nTimeslot):
            rhs_array[t] = 0.1*Pbid[t]
            if self.is_dg_reserve:
                rhs_array[t] += Rdg[t] #self.test_const # + Pbid[t]*0.1
            if self.is_ess_reserve:
                rhs_array[t] += Ress[t] #self.test_const # + Pbid[t]*0.1

            for n in range(self.nScen):
                lhs_array[t,n] = - profile_xi[t,n]
            
                if lhs_array[t,n] <= rhs_array[t] + 0.01:
                    check_array[t,n] = 1
                    count = count + 1
                else:
                    check_array[t,n] = 0
         
        ratio = count / (self.nTimeslot * self.nScen) 
        return lhs_array, rhs_array, check_array, ratio        
        

