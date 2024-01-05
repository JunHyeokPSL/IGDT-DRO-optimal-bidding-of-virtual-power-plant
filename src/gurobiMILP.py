# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:03:33 2023

@author: junhyeok
"""
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

class gurobi_MILP:
    def __init__(self, NAME, vpp, model_dict, case_dict):
        
        self.m = gp.Model(name=NAME)
        self.vpp = vpp
        
        self.model_dict = model_dict
        
        
        self.wt_list = self.vpp.wt_list
        self.pv_list = self.vpp.pv_list
        self.ess_list = self.vpp.ess_list
        self.dg_list = self.vpp.dg_list
        
        self.nWT = self.vpp.nWT
        self.nPV = self.vpp.nPV
        self.nESS = self.vpp.nESS        
        self.nRES = self.nWT + self.nPV
        self.nDG = vpp.nDG
        self.dayahead_smp = self.vpp.da_smp_profile
        
        self.N_PIECE = vpp.N_PIECE
        
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
            #self.delta = self.model_dict['delta']
        elif self.is_case4:
            self.is_res_var = True
            self.is_uncertainty = False
            #self.delta = self.model_dict['delta']            
            
        else:
            raise Exception("No Considered Case at init is_res_var")
    
        
        
        self.is_case_risk_averse = self.case_dict['bid_type'] == 'risk_averse'
        self.is_igdt_risk_averse = self.case_dict['bid_type'] == 'igdt_risk_averse'
        self.is_igdt_risk_seeking = self.case_dict['bid_type'] == 'igdt_risk_seeking'
        
        
        self.UNIT_TIME = self.case_dict['UNIT_TIME'] 
        self.nTimeslot = int (24 / self.UNIT_TIME)
        self.nScen = self.case_dict['N']
        
        
        self.base_obj = 0
        self.delta = 0
        
        self.check_set = {}
        
    def add_Variables(self):
        
        vpp = self.vpp

        if self.is_case1 or self.is_case2 or self.is_case3: 
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
            
            print("Assign P_dg in No uncertainty") 
            
            
            
        if self.is_case1:
            print("")
            print("case 1 add Variables")
            print("gurobi_MILP add Variables")
            print("No Uncertainty Sets in this case")
            print("")
        
        
        elif self.is_case2:
        
            # DRCC Objective Formulation
            # dual variables of Distributionally Robust Optimization
            self.s_obj = self.m.addVars(self.nScen, self.nTimeslot, lb = -1000000, ub = 1000000, vtype = gp.GRB.CONTINUOUS, name='s_obj')
            self.lambda_obj = self.m.addVars(self.nTimeslot, vtype=gp.GRB.CONTINUOUS, lb = 0.0, ub= 1000000.0, name='lambda_obj')
            self.theta = self.case_dict['theta']     
        
       
        elif self.is_case3:
            # DRCC Objective Formulation
            # dual variables of Distributionally Robust Optimization
            self.s_obj = self.m.addVars(self.nScen, self.nTimeslot, lb = -1000000, ub = 1000000, vtype = gp.GRB.CONTINUOUS, name='s_obj')
            self.lambda_obj = self.m.addVars(self.nTimeslot, vtype=gp.GRB.CONTINUOUS, lb = 0.000001, ub= 1000000.0, name='lambda_obj')
            self.theta = self.case_dict['theta'] # * self.wt_list[0].max_power    
            
            self.gamma_obj = self.m.addVars(2*self.nRES , self.nScen, self.nTimeslot, lb = 0,ub = 10000.0, vtype = gp.GRB.CONTINUOUS, name='gamma_obj')
       
        
        else:
            raise Exception("No Considered Case at add_Variables")
    
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
               
        
        if self.is_igdt_risk_averse or self.is_igdt_risk_seeking:
            
            if self.is_case3 or self.is_case4:
                print("case 3 or 4: alpha is constant")
                self.alpha = self.m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name= 'alpha')
            else:                
                raise Exception("No Considered Case at add_Variables for alpha")
        else:
            print("Does not Cosidered alpha")
                

    
    def add_bid_constraints(self):
        
        if self.is_case1 or self.is_case2 or self.is_case3: 
                
            for j in range(self.nTimeslot):
                wt_sum = gp.quicksum(self.wt_list[i].profile_mu[j] for i in range(self.nWT))
                pv_sum = gp.quicksum(self.pv_list[i].profile_mu[j] for i in range(self.nPV))
                dg_sum = gp.quicksum(self.P_dg[i, k, j] for i in range(self.nDG) for k in range(self.N_PIECE))
                ess_dis_sum = gp.quicksum(self.P_essDis[i, j] for i in range(self.nESS))
                ess_chg_sum = gp.quicksum(self.P_essChg[i, j] for i in range(self.nESS))
        
                self.m.addConstr(
                    self.Pbid[j] == wt_sum + pv_sum + dg_sum + ess_dis_sum - ess_chg_sum,
                    name=f'const_bid{j}'
                )    
                
                
                
        else:
            raise Exception("No Considered Case at Bid Constraints")
    
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
                         
    def add_res_constraints(self):
        
        if self.is_res_var:
            
            for j in range(self.nTimeslot):
                
                if self.is_case1:
                    for i in range(self.nWT):
                        self.m.addConstr(self.P_wt[i,j] == self.wt_list[i].max_power * self.wt_list[i].profile[j],
                                        name=f'const_wt{i}_{j}_profile')
    
                    for i in range(self.nPV):
                        self.m.addConstr(self.P_pv[i,j] == self.pv_list[i].max_power * self.pv_list[i].profile[j],
                                        name=f'const_pv{i}_{j}_profile')
                elif self.is_case2:
                    for i in range(self.nWT):
                        self.m.addConstr(self.P_wt[i,j] == (1 + self.P_wt_uncertainty[i,j] ) *                                       
                                         self.wt_list[i].max_power * self.wt_list[i].profile[j],
                                        name=f'const_wt{i}_{j}_profile_uncertainty')
    
                    for i in range(self.nPV):
                        self.m.addConstr(self.P_pv[i,j] ==(1 + self.P_pv_uncertainty[i,j] ) *                                           
                                         self.pv_list[i].max_power * self.pv_list[i].profile[j],
                                        name=f'const_pv{i}_{j}_profile_uncertainty')
                elif self.is_case4:
                    if self.is_igdt_risk_averse:
                        for i in range(self.nWT):
                            self.m.addConstr(self.P_wt[i,j] == (1 - self.alpha) *                                       
                                             self.wt_list[i].max_power * self.wt_list[i].profile[j],
                                            name=f'const_wt{i}_{j}_profile_uncertainty')
        
                        for i in range(self.nPV):
                            self.m.addConstr(self.P_pv[i,j] ==(1) *                                           
                                             self.pv_list[i].max_power * self.pv_list[i].profile[j],
                                            name=f'const_pv{i}_{j}_profile_uncertainty')
                        # for i in range(self.nPV):
                        #     self.m.addConstr(self.P_pv[i,j] ==(1 - self.alpha) *                                           
                        #                      self.pv_list[i].max_power * self.pv_list[i].profile[j],
                        #                     name=f'const_pv{i}_{j}_profile_uncertainty')
                            
                    if self.is_igdt_risk_seeking:
                        for i in range(self.nWT):
                            self.m.addConstr(self.P_wt[i,j] == (1 + self.alpha) *                                       
                                             self.wt_list[i].max_power * self.wt_list[i].profile[j],
                                            name=f'const_wt{i}_{j}_profile_uncertainty')
        
                        for i in range(self.nPV):
                            self.m.addConstr(self.P_pv[i,j] ==(1) *                                           
                                             self.pv_list[i].max_power * self.pv_list[i].profile[j],
                                            name=f'const_pv{i}_{j}_profile_uncertainty')  
                        # for i in range(self.nPV):
                        #     self.m.addConstr(self.P_pv[i,j] ==(1 + self.self.alpha) *                                           
                        #                      self.pv_list[i].max_power * self.pv_list[i].profile[j],
                        #                     name=f'const_pv{i}_{j}_profile_uncertainty') 
                     
                else:                   
                    raise Exception("No Considered Case at add_res_constraints")
                    
            print("add_res_constraints completed")
            
    def add_dg_constraints(self):
        
        
        for j in range(self.nTimeslot):
            
            for i in range(self.nDG):
                
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
                    
                    # SoC min max constraint
                    self.m.addConstr(ess_list[i].initSOC 
                                     - sum(self.P_essDis[i, k] * self.UNIT_TIME 
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
                                      - sum(self.P_essDis[i, k] * self.UNIT_TIME 
                                            / ess_list[i].efficiency / ess_list[i].max_capacity
                                            for k in range(self.nTimeslot))
                                      + sum(self.P_essChg[i, k] * self.UNIT_TIME 
                                            * ess_list[i].efficiency / ess_list[i].max_capacity
                                            for k in range(self.nTimeslot)) == ess_list[i].termSOC,
                                     name=f'const_ess{i}_term')  
    
    def set_igdt_params(self, base_obj, delta):
        self.base_obj = base_obj
        self.delta = delta        
    
    def add_igdt_constraints(self):
        
        if self.is_igdt_risk_averse:
            self.revenue =gp.quicksum(self.da_smp[j]*self.Pbid[j] for j in range(self.nTimeslot))
            self.m.addConstr(self.revenue >= (1-self.delta)*self.base_obj,
                             name = 'const_igdt_risk_averse_cost')
            
        elif self.is_igdt_risk_seeking:
            self.revenue =gp.quicksum(self.da_smp[j]*self.Pbid[j] for j in range(self.nTimeslot))
            self.m.addConstr(self.revenue >= (1+self.delta)*self.base_obj,
                             name = 'const_igdt_risk_seeking_cost')
        
            print("add_igdt_risk_seeking_constraints sucessfully") 
        
    def set_DRO_obj_constraints(self):
        
        # Now consider the only WT case
        if self.is_case2:
            self.lhs_dual = self.m.addMVar((self.nTimeslot) , name = 'dro_dual_norm')
            for j in range(self.nTimeslot):
                # Constraints that c \xi <= si regardless gamma term with (h-H\xi)
                for n in range(self.nScen):
                    lhs = 0
                    for i in range(self.nWT):
                        lhs += self.dayahead_smp[j] * self.wt_list[i].profile_xi[j][n]
                    self.m.addConstr(self.s_obj[n,j] >= lhs, name = f'Const_si_dual_{j}_{n}')
                # Add Dual Norm Constraint for obj
                
                
                self.C_list = [self.dayahead_smp[j]] *self.nWT
                
                # self.m.addGenConstrNorm(self.lhs_dual[j], self.C_list, GRB.INFINITY, f"Const_norm_obj_{j}")
                #self.addConstr(self.lhs_dual[j] <= self.lambda_obj[j], f"Const_norm_obj_comp_{j}")
                self.m.addConstr( self.lambda_obj[j] >= max(self.C_list), f"Const_norm_obj_comp_{j}")
                
                
        elif self.is_case3:
            self.lhs_dual = self.m.addMVar((self.nScen, self.nTimeslot) , name = 'dro_dual_norm')
            for j in range(self.nTimeslot):                
                # Constraints that c \xi <= si regardless gamma term with (h-H\xi)
                for n in range(self.nScen):
                    lhs = 0
                    for i in range(self.nWT):
                        
                        xi_hat = self.wt_list[i].profile_xi[j][n]
                        Wmax = self.wt_list[i].max_power
                        mu_hat = self.wt_list[i].profile_mu[j]
                        lhs += self.dayahead_smp[j] * xi_hat
                        
                        #h1 = self.wt_list[i].max_power - self.wt_list[i].profile_mu[j]
                        h1 = Wmax - mu_hat
                        h2 = mu_hat
                        H1 = 1
                        H2 = -1
                        lhs += self.gamma_obj[i,n,j]*(h1 - H1 * xi_hat)
                        lhs += self.gamma_obj[i+self.nWT, n, j] * (h2 - H2 * xi_hat)                            
                        
                    self.m.addConstr( lhs <= self.s_obj[n,j] , name = f'Const_si_dual_{j}_{n}')
                    
                    # Add Dual Norm Constraint for obj
                    H1 = 1
                    H2 = -1
                    YY_list = []                    
                    for i in range(self.nWT):
                        self.YY = self.m.addVar(vtype = gp.GRB.CONTINUOUS, lb = -1000000.0, name=f'YY_{i}_{n}_{j}')
                        self.m.addConstr( self.YY == H1 * self.gamma_obj[i,n,j] + H2 * self.gamma_obj[i+self.nWT,n,j] - self.dayahead_smp[j], name=f'YYe_{i}_{n}_{j}')
                        YY_list.append(self.YY)
                    #self.C_list = [self.dayahead_smp[j]] *self.nWT
                    
                    self.m.addGenConstrNorm(self.lhs_dual[n,j], YY_list, GRB.INFINITY, f"Const_norm_obj_{n}_{j}")
                    self.m.addConstr(self.lhs_dual[n,j] <= self.lambda_obj[j], f"Const_norm_obj_comp_{n}_{j}")
                    #self.m.addConstr( self.lambda_obj[j] >= max(self.C_list), f"Const_norm_obj_comp_{j}")           
      
    def set_base_Objectives(self):
        
        if self.is_case1:
            self.obj = gp.quicksum(self.dayahead_smp[j]*self.Pbid[j] for j in range(self.nTimeslot))
                #self.obj = gp.quicksum(self.dayahead_smp[j] for j in range(self.nTimeslot))
 
        elif self.is_case2 or self.is_case3:
            self.obj1 = gp.quicksum(self.dayahead_smp[j]*self.Pbid[j] for j in range(self.nTimeslot))
            self.obj2 = gp.quicksum(self.theta[j] * self.lambda_obj[j] for j in range(self.nTimeslot))
            self.obj3 = gp.quicksum(1/self.nScen * self.s_obj[i,j] for i in range(self.nScen) for j in range(self.nTimeslot))
            self.obj = self.obj1 - self.obj2 - self.obj3
            
            self.set_DRO_obj_constraints()
        
        
        if self.dg_list:
            
            self.dg_gen_cost =  gp.quicksum(self.dg_list[i].slopes[k] * self.P_dg[i, k, j] * self.dg_list[i].fuel_cost
                                            for i in range(self.nDG) for k in range(self.N_PIECE) for j in range(self.nTimeslot))
            self.dg_start_shut_cost = gp.quicksum(self.U_dg[i, j] * self.dg_list[i].a * self.UNIT_TIME * self.dg_list[i].fuel_cost
                                       + self.SU_dg[i, j] * self.dg_list[i].start_up_cost + self.SD_dg[i, j] * self.dg_list[i].shut_down_cost for i in range(self.nDG) for j in range(self.nTimeslot))
            self.dg_obj_cost = self.dg_gen_cost + self.dg_start_shut_cost
            
            self.obj = self.obj - self.dg_obj_cost
    
    def set_Objectives(self):
        
        self.set_base_Objectives()
        self.set_obj = self.m.setObjective(self.obj, GRB.MAXIMIZE)
        
        return self.obj
    
    def solve(self, tol, timelimit=None):
        
        mip_gap = tol[0]
        feas_tol = tol[1]
        self.m.setParam(GRB.Param.MIPGap, mip_gap)
        self.m.setParam(GRB.Param.FeasibilityTol, feas_tol)
        
        time_start_op = time.time()
        sol = self.m.optimize()
        time_end_op = time.time()
        
        print("Optimization Duration Time:", time_end_op - time_start_op)
        
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
        
        P_dgSol = np.zeros([self.nDG, self.N_PIECE, self.nTimeslot])
        sum_P_dgSol = np.zeros([self.nDG, self.nTimeslot])
        U_dgSol = np.zeros([self.nDG, self.nTimeslot])
        SU_dgSol = np.zeros([self.nDG, self.nTimeslot])
        SD_dgSol = np.zeros([self.nDG, self.nTimeslot])
        

        lambda_objSol = np.zeros([self.nTimeslot])
        s_objSol = np.zeros([self.nScen, self.nTimeslot])
        sum_s_objSol = np.zeros([self.nTimeslot])
        gamma_objSol = np.zeros([2*self.nRES, self.nScen, self.nTimeslot])
        
        s_obj_exists = "s_obj[0,0]" in [var.VarName for var in self.m.getVars()]
        lambda_obj_exists = "lambda_obj[0]" in [var.VarName for var in self.m.getVars()]
        gamma_obj_exists = "gamma_obj[0,0,0]" in [var.VarName for var in self.m.getVars()] 
        YY_Sol = np.zeros([self.nWT, self.nScen, self.nTimeslot])
        
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
                    continue
                        
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
                    continue
                
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
                    continue
            except Exception as e:
                print("Error")
                print(e)
                print("gamma_obj error")
                print("gamma_obj error")
                
                    
            if self.is_case3:
                for i in range(self.nRES):
                    for n in range(self.nScen):
                        YY_Sol[i,n,j] = self.m.getVarByName(f"YY_{i}_{n}_{j}").X
            
            for i in range(self.nESS):
                P_essDisSol[i,j] = self.m.getVarByName(f"P_essDis[{i},{j}]").X
                P_essChgSol[i,j] = self.m.getVarByName(f"P_essChg[{i},{j}]").X
                U_essDisSol[i,j] = self.m.getVarByName(f"U_essDis[{i},{j}]").X
                U_essChgSol[i,j] = self.m.getVarByName(f"U_essChg[{i},{j}]").X

            for i in range(self.nDG):
                for k in range(self.N_PIECE):
                    P_dgSol[i,k,j] = self.m.getVarByName(f"P_dg[{i},{k},{j}]").X
                    sum_P_dgSol[i,j] += P_dgSol[i,k,j]
                U_dgSol[i,j] = self.m.getVarByName(f"U_dg[{i},{j}]").X 
                SU_dgSol[i,j] = self.m.getVarByName(f"SU_dg[{i},{j}]").X 
                SD_dgSol[i,j] = self.m.getVarByName(f"SD_dg[{i},{j}]").X 
                
        P_dict = {'bid': P_BidSol}
        U_dict = {}
        
        if self.dg_list:
            P_dict['dg'] = P_dgSol
            P_dict['sum_dg'] = sum_P_dgSol
            U_dict['dg'] = U_dgSol
            U_dict['dg_su'] = SU_dgSol
            U_dict['dg_sd'] = SD_dgSol
            
            
        if self.ess_list:
            P_dict['essDis'] = P_essDisSol
            P_dict['essChg'] = P_essChgSol
            U_dict['essDis'] = U_essDisSol
            U_dict['essChg'] = U_essChgSol
            
        if self.is_case2:
            U_dict['lambda_obj'] = lambda_objSol
            U_dict['s_obj'] = s_objSol 
            
        if self.is_case3:
            U_dict['lambda_obj'] = lambda_objSol
            U_dict['s_obj'] = s_objSol 
            U_dict['gamma_obj'] = gamma_objSol
            U_dict['YY'] = YY_Sol

        self.P_dict = P_dict
        self.U_dict = U_dict
        
        return P_dict, U_dict
    
    def optimize(self):
        
        self.add_Variables()
        self.add_bid_constraints()
        # self.add_smp_constraints()
        # self.add_res_constraints()
        
        self.add_dg_constraints()
        self.add_ess_contraints()
        # self.add_igdt_constraints()
        obj_eq = self.set_Objectives()
        
        mip_gap = 0.0001
        feas_tol = 1e-4
        sol, obj = self.solve([mip_gap, feas_tol])
        
        P_dict, U_dict = self.get_sol()
        
        obj_dict = {}
        
        if self.is_case1:
            PbidSol = P_dict['bid']
            
            obj1 = np.zeros(self.nTimeslot)
            for j in range(self.nTimeslot):
                obj1[j] = self.dayahead_smp[j] * PbidSol[j]
            obj_dict['obj1'] = obj1
            
            
        else:
            try:
                PbidSol = P_dict['bid']
                lambda_objSol = U_dict['lambda_obj']
                s_objSol = U_dict['s_obj']
                
                obj1 = np.zeros(self.nTimeslot)
                obj2 = np.zeros(self.nTimeslot)
                obj3 = np.zeros(self.nTimeslot)
                obj3_full = np.zeros((self.nScen, self.nTimeslot))
                
                obj_dg_gen_cost = np.zeros((self.nDG, self.N_PIECE, self.nTimeslot))
                obj_dg_sum_gen_cost = np.zeros((self.nDG, self.nTimeslot))
                
                obj_dg_run_cost = np.zeros((self.nDG, self.nTimeslot))
                obj_dg_cost = np.zeros((self.nDG, self.nTimeslot))
                
                
                for j in range(self.nTimeslot):
                    obj1[j] = self.dayahead_smp[j] * PbidSol[j]
                    obj2[j] = self.theta[j] * lambda_objSol[j]
                    
                    for i in range(self.nScen):
                        obj3_full[i,j] = 1/self.nScen * s_objSol[i,j]
                        obj3[j] += obj3_full[i,j]
                        
                        
                    for i in range(self.nDG):
                        for k in range(self.N_PIECE):
                            obj_dg_gen_cost[i,k,j] = self.dg_list[i].slopes[k] * P_dict['dg'][i,k,j] * self.dg_list[i].fuel_cost 
                            obj_dg_sum_gen_cost[i,j] += obj_dg_gen_cost[i,k,j]
                        obj_dg_run_cost[i,j] = U_dict['dg'][i,j] * self.dg_list[i].a * self.UNIT_TIME * self.dg_list[i].fuel_cost 
                        + U_dict['dg_su'][i,j] * self.dg_list[i].start_up_cost + U_dict['dg_sd'][i,j] * self.dg_list[i].shut_down_cost
                        obj_dg_cost[i,j] = obj_dg_sum_gen_cost[i,j] + obj_dg_run_cost[i,j]                       
                           
                        
                obj_dict['obj1'] = obj1 
                obj_dict['obj2'] = obj2
                obj_dict['obj3'] = obj3
                obj_dict['obj3_full'] = obj3_full
                
                obj_dict['dg_sum'] = obj_dg_sum_gen_cost
                obj_dict['dg'] = obj_dg_gen_cost
                obj_dict['dg_susd'] = obj_dg_run_cost
                obj_dict['dg_cost'] = obj_dg_cost
            
            except Exception as e:
                print("Error")
                print(e)
                print("obj_dict generate failed")
        
        
        return sol, obj_dict, P_dict, U_dict
                