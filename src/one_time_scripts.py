# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:49:19 2020

@author: ers2244
"""

import presets_lynchGYN as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    
    
def define_risk(risk_index, old_oc_df, old_ec_df, gene):
    new_risk_ec = np.zeros(len(old_ec_df))
    new_risk_oc = np.zeros(len(old_oc_df))
    i = 0
    for i in range(0, len(old_oc_df)):
        if int(risk_index) < 0:
            risk_diff = old_oc_df.loc[i, 'diff_risk_floor']
        else:
            risk_diff = old_oc_df.loc[i, 'diff_risk_ceil']
            
        old_risk = old_oc_df.loc[i, 'risk_OC']
        
        oc_new_risk = old_risk + (risk_diff * int(risk_index))
        new_risk_oc[i] = oc_new_risk if oc_new_risk > 0 else 0.0
    i = 0
    for i in range(0, len(old_ec_df)):
        if int(risk_index) < 0:
            risk_diff = old_ec_df.loc[i, 'diff_risk_floor']
        else:
            risk_diff = old_ec_df.loc[i, 'diff_risk_ceil']
            
        old_risk = old_ec_df.loc[i, 'risk_EC']
        
        ec_new_risk = old_risk + (risk_diff * int(risk_index))
        new_risk_ec[i] = ec_new_risk if ec_new_risk > 0 else 0.0
        
    return new_risk_oc, new_risk_ec

#creates a spreadsheet to set cancer risks for sensitivity analysis. only need to call once
def create_risk_spreadsheet():
    old_risk_index = np.arange(-5, 6)
    risk_index = []
    for i in old_risk_index:
        risk_index.append(str(int(i)))
    writer = pd.ExcelWriter(ps.data_repo/'cancer_risk_ranges.xlsx', 
                            engine='xlsxwriter')
    
    for gene in ps.GENES:
        i = 0
        print(gene)
        old_ec_df = pd.read_excel(ps.raw_risk_data, sheet_name = f'{gene}EC')
        risk_ec_df = pd.DataFrame(columns = risk_index, index = range(0, len(old_ec_df)))
        risk_ec_df['age'] = old_ec_df['age']
        risk_ec_df.set_index('age', inplace=True)
        
        old_oc_df = pd.read_excel(ps.raw_risk_data, sheet_name = f'{gene}OC')
        risk_oc_df = pd.DataFrame(columns = risk_index, index = range(0, len(old_oc_df)))
        risk_oc_df['age'] = old_oc_df['age']
        risk_oc_df.set_index('age', inplace=True)
        for i in range(0, len(risk_index)):
            new_risk_oc, new_risk_ec = define_risk(risk_index[i], old_oc_df, old_ec_df, gene)
            risk_oc_df[risk_index[i]] = new_risk_oc
            risk_ec_df[risk_index[i]] = new_risk_ec
            
        
        risk_ec_df.reset_index(inplace = True)
        risk_oc_df.reset_index(inplace = True)
        
        risk_oc_df.to_excel(writer, sheet_name = gene+'_OC', index= False,
                            startcol = 0)
        risk_ec_df.to_excel(writer, sheet_name = gene+'_EC', index = False,
                            startcol = 0)
    # Close the Pandas Excel writer and output the Excel file.
    writer.close()
    return risk_oc_df, risk_ec_df

def beta_dist(mean, sd):
    alpha = ((1 - mean)/sd - 1/mean)**2
    beta = alpha * (1/mean - 1)
    return alpha, beta
   
#Generates alpha and beta parameters for the variable distributions
#Just need to run once and save the dataframe with alpha and beta params
#After running, update the "params_PSA" spreadsheet in model inputs
def generate_sens_params(params = ps.PARAMS_PSA, plot_params = 'all'):
    temp_params = params.copy()
    dist_info = params.copy()
    for i in temp_params.index:
        if 'risk ac death' not in i and 'lifetime risk' not in i and 'stage dist' not in i:
            #set specific params for risk of OC post-Hyst BS
            if 'tubal' in i:
                alpha = 5
                beta = 1
            else:
                alpha, beta = beta_dist(params.loc[i, 'value'],
                                        params.loc[i, 'value']*params.loc[i, 'multiplier'])
            print(i)
            print(alpha, beta)
            test_dist = np.random.beta(alpha, beta, 1000)
            print('*'*30)
            print(i)
            print('*'*30)
            dist_info.loc[i, 'alpha'] = alpha
            dist_info.loc[i, 'beta'] = beta
            print(len(test_dist[test_dist< params.loc[i, 'low_bound']]))
            dist_info.loc[i, 'below_low_bound'] = len(test_dist[test_dist< params.loc[i, 'low_bound']])
            print(min(test_dist))
            print(len(test_dist[test_dist>params.loc[i, 'up_bound']]))
            dist_info.loc[i, 'above_up_bound'] = len(test_dist[test_dist>params.loc[i, 'up_bound']])
            dist_info.loc[i, 'max_val'] = max(test_dist)
            dist_info.loc[i, 'min_val'] = min(test_dist)
            print(max(test_dist))
            print(params.loc[i, :])
            if plot_params == 'all':
                plt.hist(test_dist)
                plt.title(i)
                plt.show()
            else:
                if i in plot_params:
                    plt.hist(test_dist)
                    plt.title(i)
                    plt.show()
    dist_info.to_csv(ps.data_repo/'psa_params_temp_v1.csv')
    return dist_info

def get_gamma_params(mean, sd):
    scale = (sd/mean)
    shape = mean/scale
    return shape, scale


#This generates distribution parameters for gamma distributions
#The resulting dataframe should be merged with the "cost" sheet in model_params
#This will likely require some manual alterations to make sure that the range of vals is wide enough
def generate_cost_sens_params(costs = ps.raw_costs):
    temp = costs.copy()
    temp = temp.set_index(temp['param'])
    for i in temp.index:
        shape, scale = get_gamma_params(temp.loc[i, 'cost'],
                                        temp.loc[i, 'cost'] * .2)
        temp.loc[i, 'gamma_shape'] = shape
        temp.loc[i, 'gamma_scale'] = scale
    temp.to_csv(ps.data_repo/'cost_params_temp.csv')
    return temp

    
#oc, ec = create_risk_spreadsheet()
#dist = generate_sens_params(plot_params=['risk oc tubal ligation'])
#print(len(dist))
#generate_cost_sens_params()


