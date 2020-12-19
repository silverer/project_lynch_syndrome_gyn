# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:57:18 2019

@author: ers2244
"""

import lynch_gyn_simulator as sim
import presets_lynchGYN as ps
import probability_functions_lynchGYN as pf
import pandas as pd
import lynch_gyn_sens as sen
import numpy as np
import multiprocessing as mp
import datetime
import post_processing as pp
import time


#multiplies the utility matrix generated in create_full_util_multiplier by the D_matrix
def apply_util_matrix(d_matrix, new_util_matrix, **kwargs):
    
    d_temp = d_matrix.loc[:, list(ps.ALL_STATES.values())]
    #d_alive used to calculate unadjusted life-years
    d_alive = d_matrix.loc[:, 'overall survival']
    util_results = new_util_matrix.multiply(d_temp)
    
    util_results['row QALYs'] = util_results.sum(axis = 1, skipna = True)
    #Sum the results of rows to get undiscounted total QALYs
    util_results['total QALYs'] = util_results['row QALYs'].cumsum(axis = 0)
    
    util_results = util_results.reset_index()
    #Apply discounting to each row
    util_results['row disc QALYs'] = util_results.apply(lambda x: pf.discount(x['row QALYs'], x['index']), axis = 1)
    
    #Sum the results of discounted QALY rows to get the total/cumulative discounted QALYs
    util_results[ps.QALY_COL] = util_results['row disc QALYs'].cumsum(axis = 0)
    util_results['row LE'] = d_alive
    util_results.loc[0, 'row LE'] *= 0.5
    
    util_results['total LE'] = util_results['row LE'].cumsum(axis = 0)
    #Add utility values to dataframe--for tracking distributions in the PSA/OWSA
    for c in util_results.columns:
        if c in ps.STATE_DICT['STATE_NAME'].to_list():
            util_results[c] = new_util_matrix.loc[1, c]
            util_results.rename(columns = {c: 'U '+c}, inplace = True)
    
    return util_results

#Calculates total costs for a given cost matrix
def get_costs(d_matrix, cost_matrix = 'none'):
    cost_results = sen.apply_cost_table(d_matrix, cost_df = cost_matrix)
    cost_results = cost_results.drop(columns=['age'])
    
    cost_results['row cost'] = cost_results.sum(axis = 1, skipna = True)
    cost_results['total cost'] = cost_results['row cost'].cumsum(axis = 0)
    cost_results = cost_results.reset_index()
    cost_results['row disc cost'] = cost_results.apply(lambda x: pf.discount(x['row cost'],
                                                                    x['index']), axis = 1)
    cost_results['total disc cost'] = cost_results['row disc cost'].cumsum(axis = 0)
    
    
    return cost_results

def iterate_over_dist_dict(df_dict, qaly_df, cost_params, 
                          changed_param = None,
                          param_value = None,
                          changed_param_type = '',
                          **kwargs):
    
    outputs = pd.DataFrame(columns = ps.RAW_OUTCOME_COLS)
    index_tracker = 0
    for k in df_dict.keys():
        d_temp = df_dict[k]
        these_utils = apply_util_matrix(d_temp, qaly_df)
        cost_df = sen.generate_cost_table(d_temp, cost_params)
        cost_results = get_costs(d_temp, cost_matrix = cost_df)
        
        if 'prob_owsa' in kwargs:
            if index_tracker == 0:
                print('prob_owsa in kwargs')
            changed_param = d_temp.loc[0, 'changed param']
            param_value = d_temp.loc[0, 'param value']
        elif 'risk_levels' in kwargs:
            if index_tracker == 0:
                print('risk levels in kwargs')
            changed_param = d_temp.loc[0, 'changed param']
            param_value = d_temp.loc[0, 'param value']
            
        
        for col in outputs.columns:
            if col in d_temp.columns and (col != 'changed param' and col != 'param value'):
                outputs.loc[index_tracker, 
                            col] = d_temp.loc[len(d_temp) - 1, col]
                
            elif col == 'changed param' and changed_param:
                outputs.loc[index_tracker, col] = changed_param_type + changed_param
                
            elif col in these_utils.columns:
                outputs.loc[index_tracker, 
                            col] = these_utils.loc[len(these_utils) - 1, col]
                            
            elif col == 'param value' and param_value:
                if 'stage dist' in changed_param:
                    val = d_temp.loc[0, col]
                    space_loc = val.find(' ')
                    val = val[1:space_loc]
                    outputs.loc[index_tracker, col] = float(val)
                else:
                    outputs.loc[index_tracker, col] = param_value
                
            elif 'cost' in col:
                outputs.loc[index_tracker, col] = cost_results.loc[len(cost_results)-1,
                                                                           col]
        index_tracker += 1
        
    return outputs
    
'''
Input: df_dict: if provided, does not run simulations
        save: bool (optional) specifying whether sim should generate .csv files or just return filename array
                defaults to False to speed up run time. Risk levels can be changed to see how it impacts results
        risk_levels: optional, if provided, generates distribution matrices for all risk levels
        cols_to_change: optional, if provided, tests new utility values for the 
                        columns in the array
                        
#Output: an dataframe with outcomes (QALYs and LE) and final distributions for each run
'''
def get_agg_utils(df_dict = 1, save = False, risk_levels = [0], col_to_change=0,
                  **kwargs):
    
    if type(df_dict) == int or type(df_dict) == float:
        #if len(risk_levels) > 1:
            #df_dict = sen.iterate_risk_levels(save = save, risk_levels = risk_levels)
        if 'prob_owsa' in kwargs: 
        #elif 'prob_owsa' in kwargs:
            #instantiates parameter dictionary with indices as keys and distributions as values
            new_param_dict = sen.set_owsa_probs()
            #iterates over optimal strategies for every new parameter value
            #df_dict = sen.iterate_new_param_val(new_param_dict)
            df_dict = sen.iterate_new_param_val(new_param_dict,
                                                iterate_all = False)
            #output_cols.append('val_type')
        else:
            df_dict = sim.iterate_strategies(save_files = save)
            
    param_value = None
    changed_param = None
    
    if type(col_to_change) != int:
        #Sets qalys for an exhaustive util threshold search
        if 'seed' in kwargs:
            seed = kwargs.get('seed')
            new_qaly_df, param_value = sen.set_new_utilities(util_to_change = col_to_change,
                                                             seed = seed)
        #changes just one qaly based on uniform distribution
        else:
            new_qaly_df, param_value = sen.set_new_utilities(util_to_change = col_to_change)
        changed_param = col_to_change
        
    else:
        #If new qaly df has been set outside (e.g., in a threshold analysis)
        #no need to generate a new one
        if 'new_qaly_df' in kwargs:
            new_qaly_df = kwargs.get('new_qaly_df')
            changed_param = kwargs.get('changed_param')
            param_value = kwargs.get('param_value')
        else:
            new_qaly_df = sen.set_new_utilities()
    
    cost_params = ps.raw_costs.copy()
    cost_params = cost_params.set_index(cost_params['param'])
    
    if 'prob_owsa' in kwargs:
        
        outputs = iterate_over_dist_dict(df_dict, new_qaly_df, cost_params,
                                         changed_param = changed_param,
                                         param_value = param_value,
                                         prob_owsa = True)
    elif len(risk_levels) > 1:
        outputs = iterate_over_dist_dict(df_dict, new_qaly_df, cost_params,
                                         changed_param = changed_param,
                                         param_value = param_value,
                                         risk_levels = True)
    else:
        outputs = iterate_over_dist_dict(df_dict, new_qaly_df, cost_params,
                                         changed_param = changed_param,
                                         param_value = param_value)
        
    outputs[ps.COST_QALY_COL] = outputs[ps.COST_COL] / outputs[ps.QALY_COL]
    return outputs


def get_icers_with_dominated(result_df_full_og, genes = ps.GENES):
    #INPUT: dataframe produced by generate_ce functions
    #OUTPUT: ce_table with dominated strategies eliminated and icers added
    all_gene_icers = pd.DataFrame()
    result_df_full = result_df_full_og.copy()
    if ps.EXCLUDE_NH:
        result_df_full = result_df_full[result_df_full['strategy'] != 'Nat Hist']
    for g in genes:
        result_df = result_df_full[result_df_full['gene'] == g]
        result_df = result_df[['gene','strategy',ps.QALY_COL, ps.COST_COL]]
        # Order input table by ascending cost
        ce_icers = result_df.sort_values(by=[ps.COST_COL])
        #placeholder to keep results from other strategies
        
    #    print(ce_icers)
        print('calculating ICERs')
        ce_icers = ce_icers.reset_index(drop = True)
        num_rows = len(ce_icers)
        row = 0
        # Eliminate strongly dominated strategies (lower qalys; higher cost)
        while row < num_rows-1:
            if(ce_icers[ps.QALY_COL][row+1] < ce_icers[ps.QALY_COL][row]):
                ce_icers = ce_icers.drop([ce_icers.index[row+1]])
                ce_icers = ce_icers.reset_index(drop = True)
                num_rows = len(ce_icers)
                row = 0
            else:
                row += 1
              
        # Initiate icers column
        ce_icers.loc[:, ps.ICER_COL] = 0
        # Calculate remaining icers and eliminate weakly dominated strategies
        if len(ce_icers) > 1:
            num_rows = len(ce_icers)
            row = 1
            while row < num_rows:
                # Calculate icers
                ce_icers.loc[ce_icers.index[row], ps.ICER_COL] = (
                        (ce_icers[ps.COST_COL][row]-ce_icers[ps.COST_COL][row-1]) / 
                        (ce_icers[ps.QALY_COL][row]-ce_icers[ps.QALY_COL][row-1]))
    #            print(ce_icers)
                # If lower qaly and higher icer, eliminate strategy
                if(ce_icers.loc[ce_icers.index[row], ps.ICER_COL] < 
                   ce_icers.loc[ce_icers.index[row-1], ps.ICER_COL]):
                    ce_icers = ce_icers.drop([ce_icers.index.values[row-1]])
                    ce_icers = ce_icers.reset_index(drop = True)
                    num_rows = len(ce_icers)
                    row = row - 1
                else:
                    row += 1
        ce_icers[ps.COST_QALY_COL] = ce_icers[ps.COST_COL] / ce_icers[ps.QALY_COL]
        all_gene_icers = all_gene_icers.append(ce_icers, ignore_index=True)
        
    all_gene_icers = all_gene_icers.drop(columns = [ps.QALY_COL, ps.COST_COL,
                                                    ps.COST_QALY_COL])
    #merge icers with old results df
    all_results = result_df_full.merge(all_gene_icers, how = 'left',
                                      on = ['gene', 'strategy'])
    
    return all_results



'''
Inputs: dictionary containing all the distribution matrices for a given gene
        column to be changed in this iteration of calculations
        seed to set distribution for utility value
Outputs: dataframe with outputs for this gene and utility value
'''
def get_agg_utils_util_thresh(df_dict, col_to_change, seed):
    
    outputs = get_agg_utils(df_dict = df_dict, col_to_change = col_to_change,
                            seed = seed)
    return outputs

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#Sets seed with psuedo-random integer to allow for multiprocessing
def get_agg_utils_PSA(df_dict_list, seed):
    
    df_cols = ps.RAW_OUTCOME_COLS
    cost_cols = ps.raw_costs['param'].to_list()
    cost_cols = ['cost ' + c for c in cost_cols]
    u_cols = ['U ' + u for u in ps.UTIL_VARS]
    
    df_cols.extend(u_cols)
    df_cols.extend(cost_cols)
    df_cols.append('key')
    outputs = pd.DataFrame(columns = df_cols)
    for param in ps.PARAMS.index:
        outputs[param] = 'none'
    start = time.time()
    df_dict = df_dict_list[0]
    df_key_list = list(df_dict.keys())
    index_tracker = 0
    
    new_df_key_list = list(chunks(df_key_list, 4*(len(ps.ALL_STRATEGIES))))
    sample_size = len(new_df_key_list)
    util_dicts = sen.set_new_utilities(sample_size = sample_size, 
                                         seed = seed)
    cost_dicts = sen.set_new_costs(sample_size = sample_size,
                                   seed = seed)
    #print('util dict length: ', len(util_dicts))
    #print('cost dict length: ', len(cost_dicts))
    
    i = 0
    #print(len(new_df_key_list))
    for i in range(0, len(new_df_key_list)):
        #print(i)
        #Sets new utilities and costs to apply to each set of 48 runs in a trial
        new_qaly_df = util_dicts[i]
        
        cost_inputs = cost_dicts[i]
        temp_costs = cost_inputs.transpose()
        for c in temp_costs.columns:
            temp_costs.rename(columns = {c: 'cost '+c}, inplace=True)
        #print(new_df_key_list[0])
        k = 0
        for k in range(0, len(new_df_key_list[0])):
            d_temp = df_dict[new_df_key_list[i][k]]
            
            these_utils = apply_util_matrix(d_temp, new_qaly_df)
            
            these_costs = sen.generate_cost_table(d_temp, cost_inputs)
            cost_results = get_costs(d_temp, cost_matrix = these_costs)
            for col in outputs:
                if col in these_utils.columns:
                    outputs.loc[index_tracker, 
                                col] = these_utils.loc[len(these_utils) - 1, col]
                    
                elif col in d_temp.columns:
                    outputs.loc[index_tracker, 
                                col] = d_temp.loc[len(d_temp) - 1, col]
                
                elif 'cost' in col:
                    if 'total' in col:
                        outputs.loc[index_tracker, 
                                    col] = cost_results.loc[len(cost_results) - 1, col]
                    else:
                        outputs.loc[index_tracker, 
                                    col] = temp_costs.loc['cost', col]
            index_tracker +=1
            
        if index_tracker % 1000 == 0:
            this_end = time.time()
            print('time to run ', index_tracker, ' samples: ', (this_end - start)/60)
                
    return outputs

#runs one-way sensitivity analysis based on new utilities defined in sens 
#separate from get_agg_utils because the main loop in the body is different
def get_agg_utils_util_owsa(simulate = False, risk_levels = [0]):
    all_cols_to_change = ps.UTIL_VARS
    #omit variables that don't apply in BC optimal strategy
    keep_arr = np.full(len(all_cols_to_change), True)
    for g in range(0, len(all_cols_to_change)):
        if 'surveillance' in all_cols_to_change[g] or 'undetected' in all_cols_to_change[g]:
            keep_arr[g] = False
            
    #select relevant columns--this just saves some time
    all_cols_to_change = all_cols_to_change[keep_arr == True]
    
    dists = ['low', 'high']
    #dist_size = 2
    all_outputs = pd.DataFrame()
    print('getting utilities')
    #get the base case optimal strategy distribution matrices
    #df_dict = sim.load_bc_files(optimal_only = True)
    df_dict = sim.load_bc_files(optimal_only = True)
    cost_params = ps.raw_costs.copy()
    cost_params = cost_params.set_index(cost_params['param'])
    #set the column to iterate over for each loop
    for i in range(0, len(all_cols_to_change)):
        print('changing col: ', all_cols_to_change[i])
        #loop through distribution of utils set in sens
        for d in dists:
            new_qaly_df, new_util = sen.set_new_utilities(choose_end = d,
                                                          util_to_change = all_cols_to_change[i])
            print(new_util)
            temp_outputs = iterate_over_dist_dict(df_dict, new_qaly_df,
                                                  cost_params,
                                                  changed_param = all_cols_to_change[i],
                                                  param_value = new_util,
                                                  changed_param_type = 'U ')
            all_outputs = all_outputs.append(temp_outputs,
                                             ignore_index = True)
    all_outputs = all_outputs.reset_index(drop = True)
    return all_outputs

#runs one-way sensitivity analysis based on new utilities defined in sens 
#separate from get_agg_utils because the main loop in the body is different
#runs one-way sensitivity analysis based on new utilities defined in sens 
#separate from get_agg_utils because the main loop in the body is different
    #add changed param type for flexibility


def get_agg_utils_cost_owsa(simulate = False, cost_df_dict = 1,
                            changed_cost_param = 'na'):
    dists = ['low_bound', 'up_bound']
    cost_params = ps.raw_costs['param'].to_list()
    all_outputs = pd.DataFrame()
    
    new_qaly_df = sen.set_new_utilities()
    print('getting utilities')
    
    if type(cost_df_dict) != int:
        df_dict = sim.load_bc_files()
        changed_param = changed_cost_param
        for key in cost_df_dict.keys():
            this_cost_df = cost_df_dict[key]
            cost_params = this_cost_df.set_index(this_cost_df['param'])
            param_value = cost_params.loc[changed_cost_param, 'cost']
            temp_outputs = iterate_over_dist_dict(df_dict, new_qaly_df, 
                                                 cost_params,
                                                 changed_param = changed_param,
                                                 param_value = param_value,
                                                 changed_param_type = 'cost ')
            all_outputs = all_outputs.append(temp_outputs, ignore_index = True)
            print(len(all_outputs))
        return all_outputs
    else:
        #get the base case optimal strategy distribution matrices           
        #df_dict = sim.load_bc_files(optimal_only = True)
        df_dict = sim.load_bc_files(optimal_only = True)
        #set the column to iterate over for each loop
        for c in cost_params:
            print('changing cost: ', c)
            for d in dists:
                cost_params = sen.set_new_costs(choose_end = d, change_cost = c)
                cost_params = cost_params.set_index(cost_params['param'])
                param_value = cost_params.loc[c, 'cost']
                temp_outputs = iterate_over_dist_dict(df_dict, new_qaly_df, 
                                                      cost_params,
                                                      changed_param = c, 
                                                      param_value = param_value,
                                                      changed_param_type = 'cost ')
                all_outputs = all_outputs.append(temp_outputs, 
                                                 ignore_index = True)
        all_outputs = all_outputs.reset_index(drop = True)
        return all_outputs


#calculates and compiles results of utility OWSA and probability OWSA 
def get_all_owsa_results():
    #changes probability values
    p_fname = f"{ps.F_NAME_DICT['OWSA_OUTS_PROBS']}{ps.icer_version}_all_strats.csv"
    
    prob_results = get_agg_utils(prob_owsa = True)
    prob_results.to_csv(ps.dump/p_fname, index = False)
    
    #prob_results = pd.read_csv(ps.dump/p_fname)
    #changes utility values
    #(different fxns bc the utility OWSA doesn't require any new simulation compared to basecase)
    u_fname = f"{ps.F_NAME_DICT['OWSA_OUTS_UTILS']}{ps.icer_version}_all_strats.csv"
    
    util_results = get_agg_utils_util_owsa()
    util_results.to_csv(ps.dump/u_fname, index = False)
    
    #util_results = pd.read_csv(ps.dump/u_fname)
    #changes costs
    c_fname = f"{ps.F_NAME_DICT['OWSA_OUTS_COSTS']}{ps.icer_version}_all_strats.csv"
    
    cost_results = get_agg_utils_cost_owsa()
    cost_results.to_csv(ps.dump/c_fname, index = False)
    
    #cost_results = pd.read_csv(ps.dump/c_fname)
    #combines results
    all_outputs = pd.concat([prob_results, util_results, cost_results])
    
    all_outputs[ps.COST_QALY_COL] = all_outputs[ps.COST_COL] / all_outputs[ps.QALY_COL]
    
    all_outputs['formatted_param'] = all_outputs['changed param'].map(ps.FORMATTED_PARAMS)
    all_outputs['pretty_strategy'] = all_outputs['strategy'].map(ps.STRATEGY_DICT)
    all_outputs.to_csv(ps.dump/f"{ps.F_NAME_DICT['OWSA_OUTS_ALL']}{ps.icer_version}_all_strats.csv", 
                                  index=False)
    
    return all_outputs


'''
FUNCTIONS TO RUN THE MODEL
'''

#default is threshold based on risk
def get_threshold_outputs(threshold_var = 'lifetime risk', results_file = 0, 
                          simulate = False, save_outs = False):
    start = time.time()
    
    if type(results_file) ==int:
        #lifetime risk gets its own function since it needs to run on multiple cores
        if threshold_var == 'lifetime risk': 
            #simulate trials if no result file is provided
            print('running trials')
            thresh_vars = ['ec lifetime risk', 'oc lifetime risk']
            
            start = time.time()
            gene_arr = []
            c_type_arr = []
            for g in ps.GENES:
                gene_arr.extend([g, g])
                c_type_arr.extend(['ec', 'oc'])
            #context manager opens and closes pool as jobs are completed
            with mp.Pool(mp.cpu_count()) as pool:
                df_pooled = pool.starmap(sen.iterate_risk_levels_mp, zip(gene_arr, 
                                                                         c_type_arr))
            print('finished running risk levels')
            print(len(df_pooled))
            df_dict = {}
            for i in range(0, len(df_pooled)):
                df_dict.update(df_pooled[i])
            end = time.time()
            print('time to run risk levels: ', (end-start)/60)
            threshold_values = np.arange(-5, 6, dtype = int)
            outputs = get_agg_utils(df_dict, risk_levels = threshold_values)
            #Save results to avoid needing re-run to fix CEA processing bugs
            outputs.to_csv(ps.dump/'all_risk_thresh_outputs.csv', index = False)
            
        elif 'cost' in threshold_var:
            new_thresh_var = threshold_var.replace('cost ', '')
            thresh_vars = [new_thresh_var]
            costs_to_test = sen.set_new_costs(thresh_params = thresh_vars)
            outputs = get_agg_utils_cost_owsa(cost_df_dict = costs_to_test,
                                                  changed_cost_param = new_thresh_var)
            threshold_values = outputs['param value'].drop_duplicates().to_list()
            thresh_vars = [threshold_var]
            print(threshold_var, ": ", threshold_values)
        #For testing utilities
        elif threshold_var.startswith('U '):
            thresh_vars = [threshold_var]
            new_thresh_var = threshold_var.replace('U ', '')
            #Unlike other util functions, set the new vals here
            #keys = new util value
            #values = df with new utility inputs
            util_dict = sen.set_new_utilities(thresh_util_to_change = new_thresh_var)
                
            df_dict = sim.load_bc_files()
            outputs = pd.DataFrame()
            #Get the agg results for each util val being tested
            for k in util_dict.keys():
                temp_outs = get_agg_utils(df_dict, new_qaly_df = util_dict[k],
                                              changed_param = threshold_var,
                                              param_value = k)
                outputs = outputs.append(temp_outs, ignore_index = True)
            threshold_values = list(util_dict.keys())
        #For testing probabilities   
        else:
            thresh_vars = [threshold_var]
            
            var_dist = sen.set_threshold_probs(threshold_var)
            df_dict = sen.iterate_new_param_val(var_dist, iterate_all = True)
            for key in df_dict.keys():
                temp = df_dict[key]
                #Fix the run with the basecase value
                #this step makes sure it's included in analysis
                #otherwise, it would just be tracker as 'na'
                if temp.loc[0, 'param value'] == 'na':
                    print(var_dist[threshold_var][0])
                    temp['param value'] = var_dist[threshold_var][0]
                    temp['changed param'] = threshold_var
                df_dict[key] = temp
                
            outputs = get_agg_utils(df_dict = df_dict, save = False, prob_owsa = True)
            threshold_values = outputs['param value'].drop_duplicates().to_list()
            print(threshold_values)
            
        end = time.time()
        print('Time to test values: ', (end-start)/60)
    else:
        outputs = pd.read_csv(ps.dump/results_file)
        
        temp_outs = outputs.loc[outputs['changed param'] != 'na', :]
        thresh_vars = temp_outs['changed param'].drop_duplicates().to_list()
        threshold_values = temp_outs['param value'].astype(float).drop_duplicates().to_list()
        if 'lifetime risk' in thresh_vars[0]:
            threshold_values.append(0.0)
    #Containers for all results   
    all_ce_results = pd.DataFrame()
    for gene in ps.GENES:
        df_gene = outputs[outputs['gene'] == gene]
        
        #because 0 will be stored as 'na', append to dataframe
        if 'lifetime risk' in thresh_vars[0]:
            #add the "null" rows to the df since "0" doesn't have a changed param (it's basecase)
            condition = ((df_gene['changed param']=='na')|
                            (pd.isnull(df_gene['changed param'])))
            temp_nulls = df_gene[condition]
            temp_nulls = temp_nulls.drop_duplicates(subset=['strategy'])
            temp_nulls = temp_nulls.reset_index(drop = True)
            temp_nulls_ec = temp_nulls.copy()
            temp_nulls_ec['changed param'] = 'ec lifetime risk'
            temp_nulls_ec['param value'] = 0
            df_gene = df_gene.append(temp_nulls_ec, ignore_index = True)
            temp_nulls_oc = temp_nulls.copy()
            temp_nulls_oc['changed param'] = 'oc lifetime risk'
            temp_nulls_oc['param value'] = 0
            df_gene = df_gene.append(temp_nulls_oc, ignore_index = True)
        
        ce_results = pd.DataFrame()
        
        for t in thresh_vars:
            temp = df_gene[df_gene['changed param']==t]
            temp = temp.loc[temp['param value'] != 'na', :]
            temp['param value'] = temp['param value'].astype(float)
            print(t)
            #for each thresh val tested, get the optimal strategy
            for i in threshold_values:
                df = temp[temp['param value'] == i]
                #if the param value is actually a string, try using that to index
                if len(df) == 0:
                    print('df length is 0')
                    df = temp[temp['param value'] == str(int(i))]
                    
                df = df.sort_values(by = ps.QALY_COL, ascending = False)
                
                df.reset_index(drop = True, inplace=True)
                
                ce_results = ce_results.append(get_icers_with_dominated(df,
                                                                        genes = [gene]),
                                                ignore_index = True)
                print(len(ce_results))
        
        all_ce_results = all_ce_results.append(ce_results, ignore_index = True)
    fname = f"threshold_icers_{threshold_var}_all_genes{ps.icer_version}.csv"
    all_ce_results.to_csv(ps.dump/fname,
                          index = False)
    
    if 'lifetime risk' in threshold_var:
        risk_types = ['ec', 'oc']
        for r in risk_types:
            temp_r = r + ' lifetime risk'
            temp = all_ce_results[all_ce_results['changed param'] == temp_r]
            temp['param value'] = temp[f"lifetime {r} risk"]
            print(len(temp))
            if r == 'oc':
                print('check')
            pp.plot_one_way_optim(temp)
    else:
        pp.plot_one_way_optim(all_ce_results)
    
    return all_ce_results



def get_util_thresholds_mp(utils_to_change = ps.UTIL_VARS, sample_size = 100):
    start = time.time()
    #get the base-case distribution matrices
    df_dict = sim.iterate_strategies(save_files = False)
    pool = mp.Pool(mp.cpu_count())
    
    #create an array of df_dicts to separate outputs by gene
    #each element corresponds to a gene
    dict_array = []
    for g in ps.GENES:
        dict_array.append({k: v for k, v in df_dict.items() if g in k})
    
    
    for k in range(0, len(utils_to_change)):
        this_util = utils_to_change[k]
        util_array = [this_util, this_util, this_util, this_util]
        print(util_array)
        print('changing util: ', utils_to_change[k])
        
        for i in range(0, sample_size):
            seed = np.random.randint(1, high = 1000000, size = 1)
            #sets the seed to be the same for all 4 genes
            seed_array = [seed, seed, seed, seed]
            #runs all 4 genes simultaneously on different cores
            #since the seeds are the same, they're all testing the same values
            all_genes = pool.starmap(get_agg_utils_util_thresh, zip(dict_array,
                                                                    util_array,
                                                                    seed_array))
            temp_outputs = pd.concat([i for i in all_genes], ignore_index=True)
            #sort the outputs by QALYs, grouped by the input value and gene
            temp_optimal = temp_outputs.sort_values([ps.QALY_COL],
                                               ascending = False).groupby(['gene', 
                                                                'param value']).head(1)
            temp_optimal.reset_index(inplace=True, drop = True)
            if i == 0 and k == 0:
                #create a dataframe with all outputs and optimal outputs
                optimal_df = temp_optimal.copy()
                all_outputs = temp_outputs.copy()
            else:
                optimal_df = pd.concat([optimal_df, temp_optimal],
                                       ignore_index = True)
                all_outputs = pd.concat([all_outputs, temp_outputs],
                                       ignore_index = True)
    end = time.time()
    print('time to get util thresholds: ', end - start)
    optimal_df.reset_index(drop = True, inplace=True)
    all_outputs.to_csv(ps.dump/f'util_thresholds_all_outputs{ps.icer_version}.csv')
    return optimal_df


def run_PSA_mp(samples, seeds):
    start = time.time()
    dt = datetime.datetime.now()
    print(dt.strftime("%Y-%m-%d %H:%M"), '    running PSA')
    
    pairs = np.empty([len(samples), 2], dtype=int)
    i = 0
    for i in range(0, len(samples)):
        pairs[i][0] = samples[i]
        pairs[i][1] = seeds[i]
    #context manager opens and closes pool as jobs are completed
    with mp.Pool(len(samples)) as pool:
        
        #run the PSA
        df_pooled = pool.starmap(sen.iterate_strategies_PSA_mp, pairs)
        #make sure the df inputs are lists to pass with the random seeds
        dfs = []
        for i in range(0, len(df_pooled)):
            dfs.append([df_pooled[i]])
            
        utils = pool.starmap(get_agg_utils_PSA, zip(dfs, seeds))   
        
    end_1 = time.time()
    dt = datetime.datetime.now()
    print(dt.strftime("%Y-%m-%d %H:%M"), 
          '    time to run samples and get utils: ', (end_1-start)/60)
       
    i = 0
    for i in range(0, len(utils)):
        if i == 0:
            all_utils = utils[i].copy()
        else:
            all_utils = all_utils.append(utils[i].copy(), ignore_index = True)
    one_seed = seeds[0]
    total_samples = sum(samples)
    
    all_fname = f'full_util_outputs_psa_{total_samples}_samples_{one_seed}.csv'
    print(all_fname)
    #save the results for every strategy and gene for the trials in the pool
    all_utils.to_csv(ps.dump_psa/all_fname,
                     index=False)
    
    return all_utils

def run_PSA_mp_dfs(samples, seeds, dists):
    start = time.time()
    dt = datetime.datetime.now()
    print(dt.strftime("%Y-%m-%d %H:%M"), '    running PSA')
    #Split the probabilities into n = len(seeds) number of subsamples
    #this will create n = len(seeds) chunks to feed into the simulator
    dist_list = np.array_split(dists, len(seeds))
    
    #Runs, for example, samples of size 2 on 8 different cores simultaneously
    #context manager opens and closes pool as jobs are completed
    with mp.Pool(len(seeds)) as pool:
        #run the PSA
        df_pooled = pool.map(sen.iterate_strategies_PSA_mp_df, dist_list)
        #make sure the df inputs are lists to pass with the random seeds
        dfs = []
        for i in range(0, len(df_pooled)):
            dfs.append([df_pooled[i]])
        utils = pool.starmap(get_agg_utils_PSA, zip(dfs, seeds))   
        
    end_1 = time.time()
    dt = datetime.datetime.now()
    print(dt.strftime("%Y-%m-%d %H:%M"), 
          '    time to run samples and get utils: ', (end_1-start)/60)
       
    i = 0
    for i in range(0, len(utils)):
        if i == 0:
            all_utils = utils[i].copy()
        else:
            all_utils = all_utils.append(utils[i].copy(), ignore_index = True)
    one_seed = seeds[0]
    total_samples = sum(samples)
    
    all_fname = f'full_util_outputs_psa_{total_samples}_samples_{one_seed}{ps.icer_version}.csv'
    print(all_fname)
    #save the results for every strategy and gene for the trials in the pool
    all_utils.to_csv(ps.dump_psa/all_fname,
                     index=False)
    
    return all_utils

def run_owsa(owsa_results = 'none'):
    start = time.time()
    if type(owsa_results) == str:
        outputs = get_all_owsa_results()
    else:
        outputs = owsa_results.copy()
    end = time.time()
    print('time to run owsa: ', end-start)
    processed_outputs = pp.process_owsa_results(outputs)
    pp.create_tornado_diagrams(processed_outputs)
    
    return

#FUNCTION FOR BASECASE ANALYSIS
#If save_dfs = false and skip_iterate = True, then loads existing outputs
def run_analysis_and_graph(col = ps.QALY_COL, save_dfs = False,
                           plot_secondary = True, skip_iterate = False):
    
    df_dict = sim.iterate_strategies(save_files = save_dfs, save_w_id = False,
                                     load_bc = skip_iterate)
    
    #Functions to plot secondary/cancer outcomes and calibration
    if plot_secondary:
        pp.plot_nat_hist_outputs(df_container = df_dict)
            
        pp.plot_cancer_inc_mort(df_dict)
        
    outputs = get_agg_utils(df_dict = df_dict, save = save_dfs)
    
    
    outputs.to_csv(ps.dump/f"{ps.F_NAME_DICT['BC_QALYs']}{ps.icer_version}.csv", index = False)
    icers = get_icers_with_dominated(outputs)
    icers.to_csv(ps.dump/f"{ps.F_NAME_DICT['BC_ICERS_W_DOM']}{ps.icer_version}.csv",
                      index = False)    
    pp.graph_eff_frontiers(icers, together = True)
    icers_formatted = pp.format_all_numbers(icers)
    icers_formatted.to_csv(ps.dump/f"{ps.F_NAME_DICT['BC_FMT_ICERS_W_DOM']}{ps.icer_version}.csv")
    #pp.plot_basecase(outputs, col)
    return icers

    


def main():
    dt = datetime.datetime.now()
    print(dt.strftime("%Y-%m-%d %H:%M"), '    in main')
    
    '''
    run_type sets the type of analysis
    base case: The main analysis
    thresh: Threshold sensitivity analyses. thresh_type specifies the type of 
            threshold analysis.
            Threshold analysis should be run from the command line.
            To change utilities, add U to the front of the variable (e.g., U HSBO)
            To change costs, add cost to the front of the variable (e.g., cost HSBO)
            For variable names, see model_inputs.xlsx
    owsa: One-way sensitivity analysis.
    PSA: Probabilistic sensitivity analysis. This should be run from the
            command line.
    '''
    run_type = 'PSA'
    thresh_type = 'risk'
    run = True
    plot = False
    
    if run_type == 'base case':
        #save dfs indicates whether the distribution matrices for 4 genes X 12 strategies should be saved
        run_analysis_and_graph(save_dfs = True, plot_secondary = True,
                               skip_iterate = False)
        
    elif run_type == 'thresh':
        if thresh_type == 'util':
            changes = ['init HSBO','init hysterectomy', 'HSBO', 'hysterectomy']
            samp_size = 500
            util_threshs = get_util_thresholds_mp(utils_to_change = changes, 
                                                  sample_size = samp_size)
            util_threshs.to_csv(ps.dump/f'util_thresholds{ps.icer_version}_{samp_size}.csv', 
                                index = False)
            print(dt.strftime("%Y-%m-%d %H:%M"), '    finished getting thresholds')
            
            if plot:
                pp.plot_util_thresholds_surgery(util_threshs)
            
        elif thresh_type == 'risk':
            if run:
                risk_threshs = get_threshold_outputs(threshold_var='lifetime risk',
                                                     simulate = True,
                                                     save_outs = False)
                
            else:
                risk_threshs = get_threshold_outputs(threshold_var='lifetime risk',
                                                     results_file = 'all_risk_thresh_outputs.csv',
                                                     simulate = False,
                                                     save_outs = False)
                
            if plot:
                risk_threshs = pd.read_csv(ps.dump/f'threshold_lifetime risk{ps.icer_version}.csv')
                pp.plot_risk_thresholds(risk_threshs)
                pp.plot_risk_thresholds_cancer_incidence(risk_threshs)
        elif 'cost' in thresh_type or 'U' in thresh_type:
            get_threshold_outputs(threshold_var = thresh_type,
                                  simulate = False,
                                  save_outs = False)
        else:
            get_threshold_outputs(threshold_var = thresh_type,
                                  simulate = True,
                                  save_outs = False)
    
    elif run_type == 'owsa':
        #owsa_results = pd.read_csv(ps.dump/f"{ps.F_NAME_DICT['OWSA_OUTS_ALL']}{ps.icer_version}.csv")
        #run_owsa(owsa_results)
        run_owsa()
        
    elif run_type == 'PSA':
        #having a sample size divisible by 8 is preferable for core distribution
        #80 * 125 = 10000 samples
        #add an extra few loops in case there are duplicate seeds
        #sample_size should be divisible by the number of CPU's
        #in each loop, n = sample_size samples will be run, but they'll be distributed
        #across n = core_num number of cores
        sample_size = 16
        loops = 120
        #loops = 105
        #loops = 128
        if mp.cpu_count() >= 32:
            core_num = 32
            print('running on 32 cores')
        else:
            core_num = 8
            print('running on 8 cores')
            
        each_sample = int(sample_size/core_num)
        samples = np.full((loops, core_num), each_sample, dtype = int)
        
        #print(samples)
        seeds = np.random.choice(2000000, size = np.shape(samples), 
                                 replace = False)
        #Generate inputs for probabilities
        all_dists = sen.generate_samples(sample_size * loops)
        dists = np.array_split(all_dists, loops)
        #dists = np.array_split(all_dists, each_sample)
        print(len(dists))
        
        all_outputs = pd.DataFrame()
        start = time.time()
        i = 0
        for i in range(0, loops):
            print(i)
            #temp_outputs = run_PSA_mp(samples[i, :], seeds[i, :])
            temp_outputs = run_PSA_mp_dfs(samples[i, :], seeds[i, :], dists[i])
            all_outputs = all_outputs.append(temp_outputs, 
                                             ignore_index=True)
            
        all_outputs = pp.create_new_id(all_outputs)
        total_sample_size = loops * sample_size
        all_out_fname = (f"{ps.F_NAME_DICT['PSA_ID']}_{total_sample_size}_"+
                            f"{ps.icer_version}.csv")
        
        all_outputs.to_csv(ps.dump_psa/all_out_fname,
                               index = False)
        
        all_outputs['strategy'] = all_outputs['strategy'].map(ps.STRATEGY_DICT)
        all_ids = all_outputs['new_id'].drop_duplicates().to_list()
        if len(all_ids) > 10000:
            small_outputs = pp.select_subsample_psa(all_outputs)
            small_fname = (f"{ps.F_NAME_DICT['PSA_ID_SUB']}"+
                              f"{ps.icer_version}.csv")
            
            small_outputs.to_csv(ps.dump_psa/small_fname,
                                 index = False)
        else:
            small_outputs = all_outputs.copy()
            
        result_types = ['icer', 'qalys']
        for r in result_types:
            pp.process_psa_results(small_outputs, result_type = r, 
                                   show_plot = False)
        end = time.time()
        print('time to run psa: ', (end-start)/60)
        
    else:
        print('Run type does not exist!')
        
        

if __name__ == "__main__":
    main()


