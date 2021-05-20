# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:02:15 2019

@author: ers2244
"""

import numpy as np
import pandas as pd
import presets_lynchGYN as ps
import scipy.stats as ss
import lynch_gyn_simulator as sim
import matplotlib.pyplot as plt
import time
import os

RISK_RANGE = np.arange(-5, 6)
SAMPLE_SIZE = 10


def beta_dist(mean, sd):
    alpha = ((1 - mean)/sd - 1/mean)**2
    beta = alpha * (1/mean - 1)
    return alpha, beta


#as in  Doubilet et al 1985
def get_sd(mean):
    sd = 1.96 * np.sqrt(((1 - mean) * mean)/100)
    return sd
    
def set_stage_dists(dist_type, local_prob):
    remainder = 1 - local_prob
    if 'oc' in dist_type:
        if 'nat' in dist_type:
            distant_prob = np.random.uniform(low = 0.5, high = 0.8)
            distant_prob *= remainder
            regional_prob = 1 - (distant_prob + local_prob)
        else:
            distant_prob = np.random.uniform(low = 0.4, high = 0.6)
            distant_prob *= remainder
            regional_prob = 1 - (distant_prob + local_prob)
    else:
        distant_prob = 0
        regional_prob = 0
    return [local_prob, regional_prob, distant_prob]

#test = set_stage_dists('oc nat hist', .07)

def set_owsa_probs(test_vars = ['all']):
    temp_dict = {}
    sample_size = 2
    params = list(ps.PARAMS.index)
    if len(test_vars) == 1:
        temp_params = params
    else:
        k = 0
        temp_params = list()
        for k in params:
            if k in test_vars:
                temp_params.append(k)
        
    for param in temp_params:
        print(param)
        if param == 'oc lifetime risk':
            d_risk_level = np.arange(-5, 0)
            d_risk_level = np.append(d_risk_level, [np.arange(1, 6)])
            temp_dict[param] = [-5, 5]
        elif param == 'ec lifetime risk':
            d_risk_level = np.arange(-5, 0)
            d_risk_level = np.append(d_risk_level, [np.arange(1, 6)])
            temp_dict[param] = [-5, 5]
        elif 'stage dist' in param:
            '''
            Set params for EC dists
            '''
            if 'ec' in param:
                temp_dist = np.zeros((sample_size, 3))
                this_stage_arr = np.random.random(3)
                this_stage_arr[0] = ps.PARAMS.loc[param, 'low_bound']
                rem = 1 - this_stage_arr[0]
                '''
                Set lower bounds for intervention (EC)
                '''
                if 'intervention' in param:
                    this_stage_arr[1] = rem * .9
                    this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                else:
                    '''
                    Set lower bounds for nat hist (EC)
                    '''
                    this_stage_arr[1] = rem * .85
                    this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                    
                temp_dist[0][:] = this_stage_arr
                print(param, ': ', this_stage_arr)    
                    
                this_stage_arr = np.random.random(3)
                this_stage_arr[0] = ps.PARAMS.loc[param, 'up_bound']
                '''
                Set upper bounds for intervention (EC)
                '''
                rem = 1 - this_stage_arr[0]
                if 'intervention' in param:
                    this_stage_arr[1] = rem * .9
                    this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                else:
                    '''
                    Set upper bounds for nat hist (EC)
                    '''
                    this_stage_arr[1] = rem * .85
                    this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                print(param, ': ', this_stage_arr)      
                temp_dist[1][:] = this_stage_arr
                temp_dict[param] = temp_dist
            else:
                '''
                Set params for OC dists
                '''
                temp_dist = np.zeros((sample_size, 3))
                this_stage_arr = np.random.random(3)
                this_stage_arr[0] = ps.PARAMS.loc[param, 'low_bound']
                rem = 1 - this_stage_arr[0]
                '''
                Set lower bounds for intervention (OC)
                '''
                if 'intervention' in param:
                    this_stage_arr[1] = rem * .4
                    this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                    
                else:
                    '''
                    Set lower bounds for nat hist (OC)
                    '''
                    this_stage_arr[1] = rem * .2
                    this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                print(param, ': ', this_stage_arr)        
                temp_dist[0][:] = this_stage_arr
                    
                this_stage_arr = np.random.random(3)
                this_stage_arr[0] = ps.PARAMS.loc[param, 'up_bound']
                rem = 1 - this_stage_arr[0]
                '''
                Set upper bounds for intervention (OC)
                '''
                if 'intervention' in param:
                    this_stage_arr[1] = rem * .25
                    this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                else:
                    '''
                    Set upper bounds for nat hist (OC)
                    '''
                    this_stage_arr[1] = rem * .4
                    this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                print(param, ': ', this_stage_arr)        
                temp_dist[1][:] = this_stage_arr
                temp_dict[param] = temp_dist
        elif 'sens' in param or 'spec' in param:
                #a nothing variable to prevent setting sens and spec for OWSA (no surveillance)
            skip = True
        else:    
            temp_dict[param] = [ps.PARAMS.loc[param, 'low_bound'],
                                    ps.PARAMS.loc[param, 'up_bound']]
            
        
    return(temp_dict)
#set_owsa_probs()

#Checks to make sure that a beta distribution doesn't include any out-of-bounds values
#This is also the function that sets the values to pull from in the PSA
def check_beta_dist(params, val, sample_size, seed):
    np.random.seed(seed)
    dist = np.random.beta(params.loc[val, 'alpha'], 
                          params.loc[val, 'beta'], sample_size)
    below_vals = len(dist[dist < params.loc[val, 'low_bound']])
    above_vals = len(dist[dist > params.loc[val, 'up_bound']])
    
    if below_vals > 0 or above_vals > 0:
        np.random.seed(seed*3)
        #Replace out-of-bound values with new values
        #Draw a much bigger random sample to avoid duplicate values as much as possible
        tmp = np.random.beta(params.loc[val, 'alpha'], 
                            params.loc[val, 'beta'], sample_size*3)
        tmp = tmp[tmp < params.loc[val, 'up_bound']]
        tmp = tmp[tmp > params.loc[val, 'low_bound']]
        new_dist = dist[dist < params.loc[val, 'up_bound']]
        new_dist = dist[dist > params.loc[val, 'low_bound']]
        
        both_dists = np.append(new_dist, tmp)
        
        if len(np.unique(both_dists)) > sample_size:
            both_dists = np.unique(both_dists)
            both_dists = np.random.choice(both_dists, sample_size, replace = False)
            return 0, 0, both_dists
        else:
            both_dists =  np.random.choice(both_dists, sample_size, replace = False)
            print(len(both_dists))
            return 0, 0, both_dists
        
    return below_vals, above_vals, dist

#Sets PSA parameters based on the param distribution info 
#Param distribution info is calculated in one_time_scripts.py            
def set_psa_params(sample_size, seed = 123):
    
    np.random.seed(seed)
    dist_dict = {}
    dist_type_dict = dict(zip(ps.PARAMS_PSA.index.to_list(),
                              ps.PARAMS_PSA.dist_type.to_list()))
    all_params = list(ps.PARAMS_PSA.index)
    #NOTE: dist_info is purely for tracking purposes-does not affect actual outputs
    dist_info = pd.DataFrame()
    #loop_tracker = None
    for val in all_params:
        #print(val)
        if 'lifetime risk' in val:
            ##sets normal distribution of risk indices
            x = np.arange(-5, 6)
            xU, xL = x + .5, x - 2
            prob = ss.norm.cdf(xU, scale = 2) - ss.norm.cdf(xL, scale = 2)
            prob = prob / prob.sum() #normalize the probabilities so their sum is 1
            risk_index = np.random.choice(x, p = prob, size = sample_size)
            
            dist_dict[val] = risk_index
                
        elif 'stage dist' in val:
            temp_dist = np.zeros((sample_size, 3))
            stage_dist = ps.PARAMS_PSA.loc[val, 'value']
            
            if 'oc' in val and 'nat hist' in val:
                multiplier = 0.2
            else:
                multiplier = 0.05
                
            alpha, beta = beta_dist(stage_dist[0],
                                    multiplier*stage_dist[0])
            i = 0
            for i in range(0, sample_size):
                this_stage_arr = np.random.random(3)
                #first, set the probability for local cancer
                this_stage_arr[0] = np.random.normal(stage_dist[0],
                                                      stage_dist[0]*multiplier)
                while (this_stage_arr[0] < ps.PARAMS_PSA.loc[val, 'low_bound'] or
                       this_stage_arr[0] > ps.PARAMS_PSA.loc[val, 'up_bound']):
                    this_stage_arr[0] = np.random.normal(stage_dist[0],
                                                      stage_dist[0]*multiplier)
                    
                #calculate the remainder to be split among regional and distant
                remainder = 1 - this_stage_arr[0]
                if 'ec' in val:
                    if 'intervention' in val:
                        #randomly choose the percent of the remainder that should -> regional (vs. distant)
                        possible_dists = [0.9, 0.95, 0.98]
                        temp_multiplier = np.random.choice(possible_dists)
                        this_stage_arr[1] = remainder * temp_multiplier
                        #distant is the remainder of the remainder
                        this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                    else:
                        #randomly choose the percent of the remainder that should -> regional (vs. distant)
                        possible_dists = [0.75, 0.8, 0.85, 0.88]
                        temp_multiplier = np.random.choice(possible_dists)
                        this_stage_arr[1] = remainder * temp_multiplier
                        #distant is the remainder of the remainder
                        this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                else:
                    if 'intervention' in val or "screening" in val:
                        #randomly choose the percent of the remainder that should -> regional (vs. distant)
                        possible_dists = [0.5, 0.55, 0.6, 0.65]
                        temp_multiplier = np.random.choice(possible_dists)
                        this_stage_arr[1] = remainder * temp_multiplier
                        #distant is the remainder of the remainder
                        this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                    else:
                        #randomly choose the percent of the remainder that should -> regional (vs. distant)
                        possible_dists = [0.3, 0.35, 0.4, 0.45]
                        temp_multiplier = np.random.choice(possible_dists)
                        this_stage_arr[1] = remainder * temp_multiplier
                        #distant is the remainder of the remainder
                        this_stage_arr[2] = 1 - sum(this_stage_arr[0:2])
                temp_dist[i][:] = this_stage_arr
                
            dist_dict[val] = temp_dist
            dist_info.loc[val, 'multiplier'] = multiplier
        else:
            if 'risk ac' in val:
                this_dist = np.full(sample_size, 1.0)
            else:
                below_vals, above_vals, this_dist = check_beta_dist(ps.PARAMS_PSA,
                                                               val, sample_size,
                                                               seed)
                
                this_dist[this_dist < ps.PARAMS_PSA.loc[val, 
                                                        'low_bound']] = ps.PARAMS_PSA.loc[val, 
                                                                        'low_bound']
                this_dist[this_dist > ps.PARAMS_PSA.loc[val, 
                                                        'up_bound']] = ps.PARAMS_PSA.loc[val, 
                                                                        'up_bound']
                
                dist_info.loc[val, 'min'] = min(this_dist)
                dist_info.loc[val, 'max'] = max(this_dist)
                
            #update the tracker
            dist_dict[val] = this_dist
            #Add info to tracking df
            dist_info.loc[val, 'reassign_upper'] = above_vals
            dist_info.loc[val, 'reassign_lower'] = below_vals
            dist_info.loc[val, 'dist_type'] = dist_type_dict[val]
    #save the tracker for this seed
    #helps to make sure the setting functions are working
    #dist_info.to_csv(ps.dump_psa/f'psa_sampling_info_{seed}.csv')
    return dist_dict
#Visualize the stage distributions set by the PSA params function
def test_stage_dists():
    dists = set_psa_params(10000)
    
    dist_types = ['oc stage dist nat hist', 'oc stage dist intervention',
                  'ec stage dist nat hist', 'ec stage dist intervention']
    
    for d in dist_types:
        temp = dists[d]
        temp_local = temp[:, 0]
        plt.hist(temp_local, label = 'local')
        temp_reg = temp[:, 1]
        plt.hist(temp_reg, label = 'regional')
        temp_distant = temp[:, 2]
        plt.hist(temp_distant, label = 'distant')
        plt.title(d)
        plt.legend()
        plt.show()
#test_stage_dists()

#plots distributions used in PSA from the processed optimal output file
def plot_dist_charts(outputs):
    df = outputs.drop_duplicates(subset = 'new_id')
    params = ps.PARAMS.index.to_list()
    for param in params:
        if 'stage dist' not in param:
            plt.hist(df[param])
            plt.title('Distribution: '+param)
            plt.tight_layout()
            file = f'histogram_inputs_{param}_{(len(df))}_samples.png'
            #plt.savefig(ps.dump/file, bbox_inches = 'tight', dpi = 300)
            plt.show()

#outs = pd.read_csv(ps.dump_psa/'PSA_optimal_mp_10000_samples_test_risk.csv')
#plot_dist_charts(outs)
#mostly for testing--shows distributions from a dictionary of distributions
#to see dists, just pass the test dict from set_psa_params to this function
def plot_dist_charts_from_dict(var_dict):
    
    params = ps.PARAMS.index.to_list()
    for param in params:
        if 'stage dist' not in param:
            plt.hist(var_dict[param])
            plt.title('Distribution: '+param)
            
            plt.tight_layout()
            plt.show()
        

def generate_samples(sample_size, seed):
    print("generating samples")
    param_dists = set_psa_params(sample_size, seed)
    print("compiling samples")
    samples_df = pd.DataFrame()
    for k in param_dists.keys():
        #The stage distributions are tempermental bc they're numpy arrays
        if 'stage dist' in k:
            samples_df[k] = None
            samples_df[k] = samples_df[k].astype(object)
            #Create new variables to concatenate into an array
            samples_df['x'] = param_dists[k][0:,0]
            samples_df['y'] = param_dists[k][0:,1]
            samples_df['z'] =  param_dists[k][0:,2]
            samples_df[k] = samples_df.apply(lambda r: tuple(r[['x', 'y', 'z']]), axis=1).apply(np.array)
            samples_df = samples_df.drop(columns = ['x', 'y', 'z'])
        else:
            samples_df[k] = param_dists[k]
    fname = ps.dump_psa/f"PSA_inputs_{sample_size}_samples_{seed[0]}{ps.icer_version}.csv"
    if os.path.exists(fname):
        fname = ps.dump_psa/f"PSA_inputs_{sample_size}_samples_{seed[0]}{ps.icer_version}_1.csv"
    
    samples_df.to_csv(fname, 
                        index = False)
    return samples_df

#plot_dist_charts_from_dict(test)

#sets thresholds for probabilities in threshold analyses
def set_threshold_probs(variable):
    if 'lifetime risk' in variable:
        return np.arange(-5, 6, dtype = int)
    dist = {}
    slices = 5
    step_size = (ps.PARAMS.loc[variable, 'up_bound'] -
                 ps.PARAMS.loc[variable, 'low_bound'])/slices
    test_range = np.arange(ps.PARAMS.loc[variable, 'low_bound'],
                           ps.PARAMS.loc[variable, 'up_bound'],
                           step_size)
    test_range = np.append(test_range, ps.PARAMS.loc[variable, 'up_bound'])
    dist[variable] = test_range
    return dist


#tests values for one parameter at a time
#returns a dictionary (instead of .csv filenames) that can be processed in ICER script
def iterate_new_param_val(new_param_dist_dict, iterate_all = False):
    new_vars = list(new_param_dist_dict.keys())
    df_dict = {}
    for var in new_vars:
        print('testing var: ', var)
        new_values = new_param_dist_dict[var]
        print(new_values)
        i = 0
        for i in range(0, len(new_values)):
            print('testing value: ', new_values[i])
            these_params = ps.PARAMS.copy()
            these_params.at[var, 'value'] = new_values[i]
            
            if iterate_all == True:
                df_dict.update(sim.iterate_strategies(save_files = False,
                                                      params = these_params))
                print(len(df_dict))
                    
                
            else:    
                df_dict.update(sim.optimal_strats_owsa(params = these_params,
                                                       save_files = False))
    return df_dict
            

#new util df has the utilities to multiply by age
#set_new_utilities generates the new_util_df that's required for this function
def create_new_util_matrix(new_util_df):
    #base_utils is an empty df to support imputing new values
    base_utils = ps.UTIL_HELPER.copy()
    new_util_df = new_util_df.reset_index(drop = True)
    stype_util_dict = {'HSBO': new_util_df.loc[0, 'HSBO'],
                       'hysterectomy': new_util_df.loc[0, 'hysterectomy']}
    #Sets u HSBO based on age
    #If age < 50, there's a decrement for early menopause
    #After and at age 50, there's no decrement for early menopause
    base_utils['HSBO'] = base_utils['surgery_util'].map(stype_util_dict)
    #Apply age weights
    base_utils['HSBO'] = base_utils['HSBO'] * base_utils['healthy']
    #Sets init HSBO based on HSBO value at a given age
    base_utils['init HSBO'] = (base_utils['HSBO'] - 
                              (new_util_df.loc[0, 'init HSBO'] * base_utils['healthy']))
    
    base_utils['surgical comps'] = (base_utils['HSBO'] - 
                                      (new_util_df.loc[0, 'surgical comps'] * base_utils['healthy']))
    
    #Apply age weights
    base_utils['hysterectomy'] = new_util_df.loc[0, 'hysterectomy'] * base_utils['healthy']
    #Sets init hysterectomy based on hysterectomy value at a given age
    base_utils['init hysterectomy'] = (base_utils['hysterectomy'] - 
                                      (new_util_df.loc[0, 'init hysterectomy'] * 
                                       base_utils['healthy']))
    
    survey_cols = ['gyn surveillance', 'undetected OC', 'undetected EC']
    
    for s in survey_cols:
        base_utils[s] = (base_utils['healthy'] - 
                          (new_util_df.loc[0, 'gyn surveillance'] * base_utils['healthy']))
    
    skip_cols = ['healthy','HSBO', 'init HSBO', 'init hysterectomy', 
                 'surgical comps','hysterectomy']
    skip_cols.extend(survey_cols)
    base_utils = base_utils.drop(columns=['surgery_util', 'age'])
    for c in base_utils.columns:
        if c not in skip_cols:
            base_utils[c] = new_util_df.loc[0, c]
            base_utils[c] = base_utils[c] * base_utils['healthy']
            
    base_utils = base_utils.fillna(0.)
    for c in base_utils.columns:
        base_utils.loc[0, c] *= 0.5
    
    
    return base_utils



#Generates a full distribution of utilities to pull from in PSA
def generate_base_util_dists(sample_size, seed = time.time()):
    
    old_utils = ps.raw_utils.copy()
    dists = pd.DataFrame(columns = ps.UTIL_VARS)
    np.random.seed(seed)
    for col in old_utils.columns:
        
        this_old_util = old_utils.loc[35, col]
        if col in ps.UTIL_VARS:
            new_utils = np.random.normal(this_old_util, this_old_util*.1, size = sample_size)
            above = new_utils[new_utils > ps.raw_utils_up.loc[35, col]]
            below = new_utils[new_utils < ps.raw_utils_low.loc[35, col]]
            #if more than 25% of values need to be dumped, sample a tighter distribution
            if len(above) > (0.25 * sample_size) or len(below) > (0.25 * sample_size):
                new_utils = np.random.normal(this_old_util, this_old_util*.05, 
                                             size = sample_size)
            
            new_utils[new_utils > ps.raw_utils_up.loc[35, col]] = this_old_util
            new_utils[new_utils < ps.raw_utils_low.loc[35, col]] = this_old_util
        else:
            new_utils = np.full(sample_size, this_old_util)
        new_utils = list(new_utils)
        dists[col] = new_utils
    fname = f"utility_dists_psa_{sample_size}_samples{ps.icer_version}.csv"
    if os.path.exists(ps.dump_psa/fname):
        rand_int = np.random.choice(200, size = 1)
        fname = f"utility_dists_psa_{sample_size}_samples{ps.icer_version}_{rand_int[0]}.csv"
    dists.to_csv(ps.dump_psa/fname, index = False)
    return dists

#test_out = generate_base_util_dists(10, 123)
#print(len(test_out))

def set_new_utilities(**kwargs):
    
    old_utils = ps.raw_utils.copy()
    new_utils = ps.raw_utils.copy()
    #Assume OWSA if no sample size provided
    if 'util_to_change' in kwargs and 'sample_size' not in kwargs:
        util_to_change = kwargs.get('util_to_change')
        
        if 'seed' in kwargs:
            np.random.seed(seed=kwargs.get('seed'))
        if 'choose_end' in kwargs:
            choose_end = kwargs.get('choose_end')
            temp_utils = ps.UTIL_ENDS[choose_end].copy()
            
            new_u = temp_utils.loc[35, util_to_change]
            new_utils.loc[35, util_to_change]  = new_u
            
        else:
            temp_u = old_utils.loc[35, util_to_change]
            new_u = np.random.uniform(low = temp_u - (temp_u * 0.1),
                                      high = temp_u + (temp_u * 0.1))
            while new_u < 0.0 or new_u > 1.0:
                new_u = np.random.uniform(low = temp_u - (temp_u * 0.1),
                                                         high = temp_u + (temp_u * 0.1))
            new_utils.loc[35, util_to_change]  = new_u
        
        new_util_matrix = create_new_util_matrix(new_utils)
        return new_util_matrix, new_u
    
    elif 'new_utilities' in kwargs:
        util_dict_matrix = {}
        util_dist = kwargs.get('new_utilities')
        util_dist = util_dist.reset_index(drop = True)
        for i in range(0, len(util_dist)):
            this_dist = util_dist.loc[i:i, :]
            new_utils = ps.raw_utils.copy()
            new_utils.loc[35, :] = this_dist.loc[i, :]
            util_dict_matrix[i] = create_new_util_matrix(new_utils)
            
        return util_dict_matrix
    
    elif 'sample_size' in kwargs:
        seed = kwargs.get('seed')
        sample_size = kwargs.get('sample_size')
        util_dist = generate_base_util_dists(sample_size, seed)
        util_dict_matrix = {}
    
        for i in range(0, sample_size):
            this_dist = util_dist.loc[i:i, :]
            new_utils = ps.raw_utils.copy()
            new_utils.loc[35, :] = this_dist.loc[i, :]
            util_dict_matrix[i] = create_new_util_matrix(new_utils)
            
        return util_dict_matrix
    
    elif 'thresh_util_to_change' in kwargs:
        util_to_change = kwargs.get('thresh_util_to_change')
        
        util_dict_matrix = {}
        
        low_bound = ps.UTIL_ENDS['low'].copy()
        low_bound = low_bound.loc[35, util_to_change]
            
        up_bound = ps.UTIL_ENDS['high'].copy()
        up_bound = up_bound.loc[35, util_to_change]
            
        slices = 10
            
        vals_to_test = np.arange(low_bound, up_bound,
                                 (up_bound-low_bound)/slices)
        
        vals_to_test = np.append(vals_to_test, up_bound)
        if old_utils.loc[35, util_to_change] not in vals_to_test:
            vals_to_test = np.append(vals_to_test, 
                                     old_utils.loc[35, util_to_change])
        print(vals_to_test)
            
        k = 0
        for k in range(0, len(vals_to_test)):
            new_utils = ps.raw_utils.copy()
            new_utils.loc[35, util_to_change] = vals_to_test[k]
            util_dict_matrix[vals_to_test[k]] = create_new_util_matrix(new_utils)
            
        return util_dict_matrix
    else:
        new_util_matrix = create_new_util_matrix(old_utils)
        return new_util_matrix
#test_1 = set_new_utilities(new_utilities = test_out)
#test = set_new_utilities(thresh_params =  ['HSBO'])
#Checks to make sure there aren't out of bound values in the gamma distribution
#Also sets the gamma dists for the PSA
def check_gamma_dist(params, val, sample_size = 10000, seed = time.time()):
    np.random.seed(seed)
    dist = np.random.gamma(params.loc[val, 'gamma_shape'], 
                          params.loc[val, 'gamma_scale'], sample_size)
    below_vals = len(dist[dist < params.loc[val, 'low_bound']])
    above_vals = len(dist[dist > params.loc[val, 'up_bound']])
    if below_vals > sample_size/4 or above_vals > sample_size/4:
        dist = np.random.gamma(params.loc[val, 'gamma_shape'], 
                                params.loc[val, 'gamma_scale'], sample_size*20)
    
    dist[dist > params.loc[val, 'up_bound']] = params.loc[val, 'cost']
    dist[dist < params.loc[val, 'low_bound']] = params.loc[val, 'cost']
    return dist

#change_df 
def generate_cost_PSA_inputs(sample_size, seed = time.time()):
    temp_costs = ps.raw_costs.copy()
    params = temp_costs['param'].to_list()
    temp_costs = temp_costs.set_index(['param'])
    #for tracking distributions
    dist_df = pd.DataFrame(columns = params, 
                            index = np.arange(0, sample_size, 
                            dtype = int))
    # dist_df = pd.DataFrame(columns = np.arange(0, sample_size, dtype = int),
    #                         index = temp_costs.index.to_list())
    for p in params:
        dist = check_gamma_dist(temp_costs, p, 
                                sample_size = sample_size,
                                seed = seed)
        dist_df[p] = dist
    fname = f"cost_dists_psa_{sample_size}_samples{ps.icer_version}.csv"
    if os.path.exists(ps.dump_psa/fname):
        rand_int = np.random.choice(200, size = 1)
        fname = f"cost_dists_psa_{sample_size}_samples{ps.icer_version}_{rand_int[0]}.csv"
    dist_df.to_csv(ps.dump_psa/fname)
    return dist_df

#test_out = generate_cost_PSA_inputs(10, 123)
#print(test_out)
def set_new_costs(**kwargs):
    #choose_end implies OWSA
    if 'choose_end' in kwargs:
        change_cost = kwargs.get('change_cost')
        choose_end = kwargs.get('choose_end')
        cost_df = ps.raw_costs.copy()
        cost_df = cost_df.set_index(cost_df['param'])
        cost_df.loc[change_cost, 'cost'] = cost_df.loc[change_cost, choose_end]
        cost_df = cost_df.reset_index(drop = True)
        cost_df = cost_df[['param', 'cost']]
        #cost_df.columns = ['param', 'cost']
        
        return cost_df
    #sets costs for PSA
    elif 'new_costs' in kwargs:
        dist_df = kwargs.get("new_costs")
        tracker = 0
        cost_dict_matrix = {}
        for i in dist_df.index:
            temp_cost_empty = ps.raw_costs.copy()
            temp_cost_empty = temp_cost_empty.set_index(['param'])

            temp_cost_empty['cost'] = dist_df.loc[i, :]
            cost_dict_matrix[tracker] = temp_cost_empty
            tracker += 1
        
        return cost_dict_matrix
    #sets costs for PSA
    elif 'seed' in kwargs:
        sample_size = kwargs.get('sample_size')
        seed = kwargs.get('seed')
        temp_costs = ps.raw_costs.copy()
        params = temp_costs['param'].to_list()
        temp_costs = temp_costs.set_index(['param'])
        #for tracking distributions
        dist_df = pd.DataFrame(columns = np.arange(0, sample_size, dtype = int),
                               index = temp_costs.index.to_list())
        for p in params:
            dist = check_gamma_dist(temp_costs, p, 
                                    sample_size = sample_size,
                                    seed = seed)
            dist_df.loc[p, :] = dist
        
        cost_dict_matrix = {}
        for i in range(0, sample_size):
            temp_cost_empty = ps.raw_costs.copy()
            temp_cost_empty = temp_cost_empty.set_index(['param'])
            temp_cost_empty['cost'] = dist_df.loc[:, i]
            cost_dict_matrix[i] = temp_cost_empty
        
        return cost_dict_matrix
    else:
        cost_dict_matrix = {}
        temp_costs = ps.raw_costs.copy()
        params = temp_costs['param'].to_list()
        temp_costs = temp_costs.set_index(['param'])
        thresh_params = kwargs.get('thresh_params')
        for t in thresh_params:
            low_bound = temp_costs.loc[t, 'low_bound']
            up_bound = temp_costs.loc[t, 'up_bound']
            slices = 10
            test_range = np.arange(low_bound, up_bound,
                                   (up_bound-low_bound)/slices)
            test_range = np.append(test_range, up_bound)
            
            for r in test_range:
                temp_costs_empty = ps.raw_costs.copy()
                temp_costs_empty = temp_costs_empty.set_index(['param'])
                temp_costs_empty.loc[t, 'cost'] = r
                temp_costs_empty = temp_costs_empty.reset_index()
                final_costs = temp_costs_empty[['param', 'cost']]
                cost_dict_matrix[r] = final_costs
        return cost_dict_matrix
#test = set_new_costs(new_costs = test_out)
#print(test)
#Sets up the table that will be multiplied with the dist matrix from simulation
#Optional cost df parameter specifies the costs to be used for calculations
def generate_cost_table(orig_dmat, cost_df = 'none'):
    age_survey = orig_dmat.loc[0, 'age survey']
    
    age_hyst = orig_dmat.loc[0, 'age hysterectomy']
    age_ooph = orig_dmat.loc[0, 'age oophorectomy']
    
    if type(cost_df) == str:
        cost_df = ps.raw_costs.copy()
        cost_df = cost_df.set_index(cost_df['param'])
    
    
    old_costs = ps.blank_costs.copy()
    new_costs = old_costs.set_index(old_costs['age'])
    
    #Get the cost of each OC and EC alive state
    for c in new_costs.columns:
        if 'local' in c or 'regional' in c or 'distant' in c:
            new_costs[c] = cost_df.loc[c, 'cost']
            
    new_costs['new OC death'] = cost_df.loc['end OC', 'cost']
    new_costs['new EC death'] = cost_df.loc['end EC', 'cost']
    
    new_costs['gyn surveillance'] = cost_df.loc['surveillance', 'cost']
    new_costs['undetected OC'] = cost_df.loc['surveillance', 'cost']
    new_costs['undetected EC'] = cost_df.loc['surveillance', 'cost']
    new_costs['init HSBO'] = cost_df.loc['HSBO', 'cost']
    new_costs['surgical comps'] = (cost_df.loc['HSBO', 'cost'] +
                                     cost_df.loc['surgical comps', 'cost'])
    new_costs['init hysterectomy'] = cost_df.loc['HSBO', 'cost']
    
    if type(age_hyst) != str:
        #if there will be 2 surgeries, apply the reduced ooph. cost
        new_costs.loc[age_ooph:, 'surgical comps'] = (cost_df.loc['surgical comps', 'cost'] +
                                                     cost_df.loc['oophorectomy', 'cost'])
        new_costs.loc[age_ooph:, 'init HSBO'] = cost_df.loc['oophorectomy', 'cost']
    if type(age_survey) != str:
        new_costs.loc[age_survey, 'gyn surveillance'] = cost_df.loc['init surveillance',
                                                                     'cost']
    
    
    return new_costs

#Multiplies cost table with distribution matrix for summary calculations in ICER script
def apply_cost_table(orig_dmat, cost_df = 'none'):
    if type(cost_df) == str:
        cost_df = generate_cost_table(orig_dmat)
        
    cost_df = cost_df.reset_index(drop = True)
    cost_cols = cost_df.columns.to_list()
    temp_dmat = orig_dmat[cost_cols]
    cost_data = temp_dmat.multiply(cost_df)
    cost_data['age'] = temp_dmat['age']
    
    
    return cost_data
    


# #this is a separate function bc it's only informative when testing all strategies (e.g., threshold, PSA)
# def test_surveillance_performance(sample_size):
#     start = time.time()
#     d_ec_spec = np.random.uniform(low = ps.PARAMS.loc['spec endo surv', 'low_bound'],
#                                   high = ps.PARAMS.loc['spec endo surv', 'up_bound'],
#                                   size = sample_size)
#     d_ec_sens = np.random.uniform(low = ps.PARAMS.loc['sens endo surv', 'low_bound'],
#                                   high = ps.PARAMS.loc['sens endo surv', 'up_bound'],
#                                   size = sample_size)
    
#     d_oc_spec = np.random.uniform(low = ps.PARAMS.loc['spec oc surv', 'low_bound'],
#                                   high = ps.PARAMS.loc['spec oc surv', 'up_bound'],
#                                   size = sample_size)
#     d_oc_sens = np.random.uniform(low = ps.PARAMS.loc['sens oc surv', 'low_bound'],
#                                   high = ps.PARAMS.loc['sens oc surv', 'up_bound'],
#                                   size = sample_size)
    
#     perf_char_dist = {'sens endo surv': np.sort(d_ec_sens),
#                        'spec endo surv': np.sort(d_ec_spec),
#                        'sens oc surv': np.sort(d_oc_sens),
#                        'spec oc surv': np.sort(d_oc_spec)}
    
#     df_dict = {}
    
#     for key in perf_char_dist.keys():
#         these_params = ps.PARAMS.copy()
#         i = 0
#         for i in range(0, sample_size):
#             print(i)
#             if i == sample_size:
#                 print('error')
#             these_params.loc[key, 'value'] = perf_char_dist[key][i]
#             df_dict.update(sim.iterate_strategies_owsa(params = these_params, sens_type = 'survey'))
            
#     end = time.time()
#     print('run time: ', end - start)
#     return df_dict


    
#create_risk_spreadsheet()

#walks through a directory to get all files in a folder
def load_threshold_files(folder):
    df_dict = {}
    import os
    for (dirpath, dirnames, filenames) in os.walk(ps.dump/folder):
        for f in filenames:
            df_dict[f] = pd.read_csv(os.sep.join([dirpath, f]))
            
    return df_dict


#Generates distribution matrices for incremental risk levels
def iterate_risk_levels_mp(gene, cancer_type):
    print('iterating risk levels for: ', gene)
    cancer_types = [cancer_type]
    df_dict = {}
    for ca in cancer_types:
        print(f'testing {ca} lifetime risk')
        var_to_change = ca + ' lifetime risk'
        for i in range(-5, 6):
            print(i)
            new_params = ps.PARAMS.copy()
            new_params.loc[var_to_change, 'value'] = i
            
            df_dict.update(sim.iterate_strategies(genes = [gene],
                                                  params = new_params,
                                                  save_files = False,
                                                  risk_levels = True))
            
    return df_dict

def iterate_strategies_PSA_mp_df(dist):
    #print('running sample: ', seed)
    
    #set up the distribution of parameters to be tested for this seed
    
    #each element in range(0, sample_size) refers to a preset distribution sample to pull from
    i = 0
    for i in range(0, len(dist)):
        this_dist = dist.iloc[i, :]
        param_df = pd.DataFrame(this_dist)
        #param_df = param_df.transpose()
        param_df.columns = ['value']
        #print(param_df)
        if i == 0:
            #container for distribution matrices
            df_dict = sim.iterate_strategies(params = param_df, run_tracker = i,
                                                  PSA = True)
        else:
            df_dict.update(sim.iterate_strategies(params = param_df,
                                                       run_tracker = i,
                                                       PSA = True))
        
    return df_dict



