# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:08:29 2019

@author: ers2244
"""

import pandas as pd
import openpyxl
import numpy as np
import simulator_helper_fxns as fxns
import probability_functions_lynchGYN as pf
import data_manipulation as dm
import presets_lynchGYN as ps


'''

FUNCTIONS TO RUN MODEL
'''

def create_t_matrix(run_spec, params = ps.PARAMS, time = ps.time, 
                    age_0 = ps.START_AGE):
# =============================================================================
#     Creates a transition matrix for the model
#     Inputs are states of the model dictionary, connectivity dictionary,
#     time, and the guidelines object to specify run type
# =============================================================================
    
    # creates connectivity matrix
    temp_age_survey = run_spec.age_survey - 1
    temp_age_HSBO = run_spec.age_HSBO - 1
    c_matrix = fxns.dict_to_connect_matrix()
    AC_DEATH_STATES = ['healthy', 'gyn surveillance', 'undetected OC',
                       'undetected EC']
    
    EVENTS = fxns.build_event_df(run_spec, params)
    EVENTS = EVENTS.set_index(EVENTS['age'])
    if run_spec.hysterectomy_alone == True:
        temp_age_hysterectomy = run_spec.age_hysterectomy - 1
        temp_age_oophorectomy = run_spec.age_oophorectomy - 1
        temp_age_HSBO = 80
        temp_age_survey = 80
    else:
        temp_age_hysterectomy = 80
        temp_age_oophorectomy = 80
        
    state_list = ps.STATE_DICT['STATE_NAME'].values
    stages = [' local', ' regional', ' distant']
    ec_stages = ['EC' + s for s in stages]
    oc_stages = ['OC' + s for s in stages]
    
    init_ec_stages = ['init EC' + s for s in stages]
    init_oc_stages = ['init OC' + s for s in stages]
    rand_t_matrix = np.full((len(state_list), len(state_list)), 1.0)
    
    # to make code more readable
    names = dm.flip(ps.ALL_STATES)
    
    #set cancer 5-year survival probabilities
    oc_stage_surv, ec_stage_surv = fxns.get_cancer_survival(params)
    
    init_dx_states = [names[s] for s in state_list if 'init EC' in s or 'init OC' in s]
    death_states = [names[s] for s in state_list if 'death' in s]
    cancer_stages = oc_stages.copy()
    cancer_stages.extend(ec_stages)
    
    dx_states = [names[c] for c in cancer_stages]
    #convert 5-year survival to annual
    oc_stage_surv['annual_death_rate'] = 0.0
    for i in oc_stage_surv.index:
        oc_stage_surv.loc[i, 'annual_death_rate'] = pf.prob_to_rate((1 - oc_stage_surv.loc[i, 'five_year_OS']), 5)
        oc_stage_surv.loc[i, 'annual_death_prob'] = pf.rate_to_prob((oc_stage_surv.loc[i, 'annual_death_rate']), 1)
    
    ec_stage_surv['annual_death_rate'] = 0.0
    for i in ec_stage_surv.index:
        ec_stage_surv.loc[i, 'annual_death_rate'] = pf.prob_to_rate((1 - ec_stage_surv.loc[i, 'five_year_OS']), 5)
        ec_stage_surv.loc[i, 'annual_death_prob'] = pf.rate_to_prob((ec_stage_surv.loc[i, 'annual_death_rate']), 1)
    
    for t in time:
        # define risk_probs
        age = t + age_0
        oc_stage_type = EVENTS.loc[age, 'oc stage dist type']
        
        ec_stage_type = EVENTS.loc[age, 'ec stage dist type']
        
        HSBO_death_risk = EVENTS.loc[age, 'HSBO_death_risk']
        
        death_risk = ps.life_table_p.loc[age, 'p_death']  
        death_rate = ps.life_table_r.loc[age, 'r_death']
        
        death_rate_post_HSBO = death_rate * EVENTS.loc[age, 'risk ac death oc surg']
        death_risk_post_HSBO = pf.rate_to_prob(death_rate_post_HSBO, 1)
        
        HSBO_comp_risk = EVENTS.loc[age, 'HSBO_comp_risk']
        
        
        '''
        Start building t-matrix
        '''
        if t == 0:
            temp = np.array(rand_t_matrix*c_matrix)
        else:
            temp = temp
        for a in AC_DEATH_STATES:
            temp[names[a], names['AC death']] = death_risk
        '''
        #if we're in surveillance time frame but not at HSBO for all
        '''
        if age >= temp_age_survey and age < temp_age_HSBO:
            #Risk of complications and surgical mort only apply to the new HSBO people
            HSBO_death_risk_surv = HSBO_death_risk * (EVENTS.loc[age, 'new_HSBO_oc'] 
                                                        + EVENTS.loc[age, 'new_HSBO_ec'])
            
            HSBO_comp_risk_surv = HSBO_comp_risk * (EVENTS.loc[age, 'new_HSBO_oc'] +
                                                    EVENTS.loc[age, 'new_HSBO_ec'])
            '''
            If this is first year of surveillance, positive resuls go from healthy -> new state
            Else, positive results go from surveillance -> new state 
            '''
            if age == temp_age_survey:
                change_var = 'healthy'
            #if this is not the first surveillance year, positive results go from gyn surveillance
            else:
                change_var = 'gyn surveillance'
            
            #Account for those going to comps/mort here since HSBO only applies to subset
            temp[names[change_var],
                 names['init HSBO']] = ((EVENTS.loc[age, 'new_HSBO_oc'] + EVENTS.loc[age, 'new_HSBO_ec']) - 
                                             (HSBO_death_risk_surv + HSBO_comp_risk_surv))
                
            temp[names[change_var], names['HSBO death comps']] = HSBO_death_risk_surv
            temp[names[change_var], names['surgical comps']] = HSBO_comp_risk_surv
            #move undetected people to undetected state
            temp[names[change_var], 
                 names['undetected OC']] = (EVENTS.loc[age, 'undetected_oc'] - 
                                                 death_risk * EVENTS.loc[age,
                                                                         'undetected_oc'])
            temp[names[change_var], 
                 names['undetected EC']] = (EVENTS.loc[age, 'undetected_ec'] - 
                                                       death_risk * EVENTS.loc[age, 
                                                                               'undetected_ec'])
            oc_stage_dist_nh = params.loc['oc stage dist nat hist', 'value']
            ec_stage_dist_nh = params.loc['ec stage dist nat hist', 'value']
            '''
            if true positive from surveillance, move to the cancer state
            if undetected from last cycle, move them to the new cancer state
            '''
            for e in range(0, len(init_ec_stages)):
                temp[names[change_var],
                     names[init_ec_stages[e]]] = (EVENTS.loc[age, 'detected_ec'] *
                                                         params.loc[ec_stage_type, 'value'][e])
                temp[names['undetected EC'],
                     names[init_ec_stages[e]]] = ec_stage_dist_nh[e] - (death_risk * 
                                                                     ec_stage_dist_nh[e])
            for o in range(0, len(init_oc_stages)):
                temp[names[change_var],
                     names[init_oc_stages[o]]] = (EVENTS.loc[age, 'detected_oc'] *
                                                     params.loc[oc_stage_type, 'value'][o])
                temp[names['undetected OC'],
                     names[init_oc_stages[o]]] = oc_stage_dist_nh[o] - (death_risk * 
                                                                     oc_stage_dist_nh[o])
           
                
            #temp[names[change_var], names['AC death']] = death_risk
            temp[names['undetected OC'], names['undetected OC']] = 0.
            temp[names['undetected EC'], names['undetected EC']] = 0.
            temp[names['init HSBO'], names['AC death HSBO']] = death_risk_post_HSBO
            temp[names['HSBO'], names['AC death HSBO']] = death_risk_post_HSBO
            temp[names['surgical comps'], names['AC death HSBO']] = death_risk_post_HSBO
            if change_var == 'healthy':
                #healthy can't go to healthy 
                temp[names['healthy'], names['healthy']] = 0.0
                pf.normalize_switch(temp[names['healthy']], names['healthy'],
                                            names['gyn surveillance'])
            else:
                pf.normalize(temp[names['healthy']], names['healthy'])
                pf.normalize_switch(temp[names['surgical comps']], names['surgical comps'],
                                        names['HSBO'])
                pf.normalize_switch(temp[names['init HSBO']], names['init HSBO'],
                                        names['HSBO'])
                
        elif age == temp_age_HSBO or age == temp_age_hysterectomy:
            '''
            If this is the first HSBO or hysterectomy year
            '''
            #if we've had some surveillance and reached HSBO age
            if age > temp_age_survey:
                change_col = 'gyn surveillance'
                #survey -> undetected is 0 because everyone not dx w/cancer will go to surgery
                temp[names['gyn surveillance'],
                      names['undetected OC']] = 0.
                temp[names['gyn surveillance'],
                     names['undetected EC']] = 0.
            else:
                change_col = 'healthy'
                
            oc_stage_dist_nh = params.loc['oc stage dist nat hist', 'value']
            ec_stage_dist_nh = params.loc['ec stage dist nat hist', 'value']
            
            temp[names[change_col],
                         names[change_col]] = 0.0
            temp[names[change_col],
                     names['surgical comps']] = HSBO_comp_risk
                 
            for e in range(0, len(init_ec_stages)):
                temp[names[change_col],
                     names[init_ec_stages[e]]] = (EVENTS.loc[age, 'ec risk'] *
                                             params.loc[ec_stage_type, 'value'][e])
                temp[names['undetected EC'],
                     names[init_ec_stages[e]]] = ec_stage_dist_nh[e] - (death_risk * 
                                                                     ec_stage_dist_nh[e])
                    
            #print(params.loc[oc_stage_type, 'value'])
            for o in range(0, len(init_oc_stages)):
                temp[names[change_col],
                     names[init_oc_stages[o]]] = (EVENTS.loc[age, 'oc risk'] *
                                             params.loc[oc_stage_type, 'value'][o])
                temp[names['undetected OC'],
                     names[init_oc_stages[o]]] = oc_stage_dist_nh[o] - (death_risk * 
                                                                     oc_stage_dist_nh[o])
            
                #apply all-cause mortality
            temp[names[change_col], names['AC death']] = death_risk
                
            if change_col == 'gyn surveillance':
                #only option is to HSBO since there's no survey+delayed strategy
                temp[names['HSBO'], names['AC death HSBO']] = death_risk_post_HSBO
                temp[names['init HSBO'], names['AC death HSBO']] = death_risk_post_HSBO
                temp[names['surgical comps'], names['AC death HSBO']] = death_risk_post_HSBO
                #there is a risk of death for those w HystBso
                temp[names[change_col],
                     names['HSBO death comps']] = HSBO_death_risk
                
                pf.normalize_switch(temp[names['init HSBO']], names['init HSBO'],
                                    names['HSBO'])
                pf.normalize_switch(temp[names['gyn surveillance']], names['gyn surveillance'],
                                    names['init HSBO'])
                pf.normalize_switch(temp[names['surgical comps']], names['surgical comps'],
                                        names['HSBO'])
            else:
                '''
                Checks whether surgery will be hysterectomy or hyst-BSO
                '''
                if age == temp_age_hysterectomy:
                    
                    temp[names['healthy'],
                         names['hysterectomy death comps']] = HSBO_death_risk
                    
                    pf.normalize_switch(temp[names['healthy']], names['healthy'],
                                        names['init hysterectomy'])
                else:
                    
                    temp[names['healthy'],
                         names['HSBO death comps']] = HSBO_death_risk
                    
                    pf.normalize_switch(temp[names['healthy']], names['healthy'],
                                        names['init HSBO'])
                
            
        #if we've already had hysterectomy
        elif age > temp_age_hysterectomy:
            '''
            Hysterectomy has already been done
            '''
            #normalize these columns so they don't throw other normalizations off
            norm_cols = ['gyn surveillance', 'undetected OC', 'undetected EC',
                         'healthy']
            
            for n in norm_cols:
                pf.normalize(temp[names[n]], names[n])
            
            '''
            If age is still not at the age set for oophorectomy
            '''
            if age < temp_age_oophorectomy:
                pf.normalize(temp[names['HSBO']], names['HSBO'])
                pf.normalize(temp[names['init HSBO']], names['init HSBO'])
                temp[names['hysterectomy'], names['init HSBO']] = 0.0
                
                if age == temp_age_hysterectomy + 1:
                    change_var = 'init hysterectomy'
                else:
                    change_var = 'hysterectomy'
                
                
                temp[names[change_var], 
                     names['AC death hysterectomy']] = death_risk
                #OC risk will still be present, but attenuated
                for o in range(0, len(init_oc_stages)):
                    temp[names[change_var], 
                             names[init_oc_stages[o]]] = (EVENTS.loc[age, 'oc risk'] * 
                                                     params.loc[oc_stage_type, 'value'][o])
                
                if change_var == 'init hysterectomy':
                    pf.normalize_switch(temp[names['init hysterectomy']], 
                                        names['init hysterectomy'],
                                        names['hysterectomy'])
                    pf.normalize_switch(temp[names['surgical comps']], 
                                        names['surgical comps'],
                                        names['hysterectomy'])
            #if it's time for oophorectomy
            elif age == temp_age_oophorectomy:
                '''
                If it is time for oophorectomy
                '''
                #ec risk prob should = 0
                #still at risk of OC, but it attenuated
                for o in range(0, len(init_oc_stages)):
                    temp[names['hysterectomy'], 
                         names[init_oc_stages[o]]] = (EVENTS.loc[age, 'oc risk'] * 
                                                 params.loc[oc_stage_type, 'value'][o])
                        
                temp[names['hysterectomy'], 
                     names['AC death hysterectomy']] = death_risk
                temp[names['hysterectomy'],
                     names['surgical comps']] = HSBO_comp_risk
                #everyone will be going to post-oophorectomy state
                temp[names['hysterectomy'],
                     names['hysterectomy']] = 0.0
                     
                temp[names['hysterectomy'],
                     names['HSBO death comps']] = HSBO_death_risk
                pf.normalize_switch(temp[names['hysterectomy']], 
                                    names['hysterectomy'],
                                    names['init HSBO'])
            else:
                pf.normalize(temp[names['hysterectomy']], names['hysterectomy'])
                #if there's folks in init HSBO, move them to continued state
                if age == temp_age_oophorectomy + 1:
                    temp[names['init HSBO'], 
                         names['AC death HSBO']] = death_risk
                    pf.normalize_switch(temp[names['init HSBO']], 
                                        names['init HSBO'], names['HSBO'])
                    pf.normalize_switch(temp[names['surgical comps']],
                                        names['surgical comps'],
                                        names['HSBO'])
                else:
                    temp[names['HSBO'], names['AC death HSBO']] = death_risk
        
        #if we're past age at HSBO
        elif age > temp_age_HSBO:
            
            norm_cols = ['gyn surveillance', 'undetected OC', 'undetected EC',
                         'healthy', 'hysterectomy', 'init hysterectomy']
            for n in norm_cols:
                pf.normalize(temp[names[n]], names[n])
                
            #shift everyone from initial HSBO to post
            if age == temp_age_HSBO + 1:
                temp[names['init HSBO'], 
                     names['AC death HSBO']] = death_risk_post_HSBO
                temp[names['HSBO'], 
                     names['AC death HSBO']] = death_risk_post_HSBO
                temp[names['surgical comps'],
                     names['AC death HSBO']] = death_risk_post_HSBO
                pf.normalize_switch(temp[names['init HSBO']], 
                                    names['init HSBO'], names['HSBO'])
                pf.normalize_switch(temp[names['surgical comps']],
                                    names['surgical comps'], names['HSBO'])
            #or just calculate the risk of death if it's post-initial HSBO year
            else:
                temp[names['HSBO'], 
                     names['AC death HSBO']] = death_risk_post_HSBO
                pf.normalize(temp[names['init HSBO']], names['init HSBO'])
        
        #if we're in natural history mode or before age at survey and age at HSBO           
        else:
            norm_cols = ['gyn surveillance', 'undetected OC', 'undetected EC',
                         'healthy', 'hysterectomy', 'init hysterectomy',
                         'HSBO', 'init HSBO']
            for n in norm_cols:
                pf.normalize(temp[names[n]], names[n])
                 
            for e in range(0, len(init_ec_stages)):
                temp[names['healthy'],
                     names[init_ec_stages[e]]] = (EVENTS.loc[age, 'ec risk'] *
                                             params.loc[ec_stage_type, 'value'][e])
                    
            for o in range(0, len(init_oc_stages)):
                temp[names['healthy'],
                     names[init_oc_stages[o]]] = (EVENTS.loc[age, 'oc risk'] *
                                             params.loc[oc_stage_type, 'value'][o])
                    
            temp[names['healthy'],
                 names['HSBO death comps']] = HSBO_death_risk
                 
            temp[names['healthy'], names['AC death']] = death_risk
            
        '''
        The following lines execute regardless of age
        '''
        for stage in oc_stage_surv.index:
            c_death_rate = oc_stage_surv.loc[stage, 'annual_death_rate']
            c_death_rate -= ps.life_table_r.loc[age, 'r_death']
            #temp[names[stage], names['OC death']] = pf.rate_to_prob(c_death_rate, 1)
            temp[names[stage], names['OC death']] = oc_stage_surv.loc[stage, 'annual_death_prob']
            temp[names[stage], names['AC death OC']] = death_risk
            
            if names[stage] in init_dx_states:
                new_stage = stage.replace('init ', '')
                #print(stage, new_stage)
                pf.normalize_switch(temp[names[stage]], names[stage],
                                    names[new_stage])
            else:
                pf.normalize_new(temp[names[stage]], names[stage])
            
        for stage in ec_stage_surv.index:
            c_death_rate = ec_stage_surv.loc[stage, 'annual_death_rate']
            c_death_rate -= ps.life_table_r.loc[age, 'r_death']
            #temp[names[stage], names['EC death']] = pf.rate_to_prob(c_death_rate, 1)
            temp[names[stage], names['EC death']] = ec_stage_surv.loc[stage, 'annual_death_prob']
            temp[names[stage], names['AC death EC']] = death_risk
            if names[stage] in init_dx_states:
                new_stage = stage.replace('init ', '')
                pf.normalize_switch(temp[names[stage]], names[stage],
                                    names[new_stage])
            else:
                pf.normalize_new(temp[names[stage]], names[stage])
                
        '''
        Double checks distribution to make sure it sums to 1 w/o negative values
        '''
        for row in range(0, death_states[0]):
            if sum(temp[row]) < 0.9999999 or sum(temp[row]) > 1.000001:
                pf.normalize_new(temp[row], row)
                #print('normalizing row', row)
            if np.any(temp[row] < 0):
                print('error, something less than 0')
            if pf.normalize_checker(temp[row], row) == False:
                print('normalization failed', row)
            if np.any(temp[row] > 1):
                print('error, something greater than 1')
            if sum(temp[row]) < 0.99999999 or sum(temp[row]) > 1.00001:
                print(names[row])
                print('error, row does not sum to 1')
        
        #normalizes death state rows such that all transitions other than same -> same == 0
        for row in death_states:
            pf.normalize(temp[row], row)
            if np.any(temp[row] < 0):
                print('error, something less than 0')
            if np.any(temp[row] > 1):
                print('error, something greater than 1')
            if sum(temp[row]) < 0.99999999 or sum(temp[row]) > 1.00001:
                print(names[row])
                print('error, row does not sum to 1')
        for state in dx_states:
            if temp[names['HSBO'], state] > 0.0:
                print('error HSBO -> cancer != 0')
            if temp[names['init HSBO'], state] > 0.0:
                print('error, init HSBO -> cancer != 0')
                
        for state in init_dx_states:
            if temp[state, state+3] == 0:
                print('error, prob of init to continue is 0')
                
        if age > temp_age_HSBO:
            for state in dx_states:
                if temp[names['init HSBO'], state] > 0.0:
                    print('error, init HSBO -> cancer != 0')
                if temp[names['HSBO'], state] > 0.0:
                    print('error HSBO -> cancer != 0')
                if temp[names['healthy'], state] > 0.0:
                    print('error, healthy -> cancer != 0')
                if temp[names['gyn surveillance'], state] > 0.0:
                    print('error, surveillance -> cancer != 0')
        
        # adding depth of the matrix
        # creating the base of the 3D matrix
        if t == 0:
            t_matrix = temp
        else:
            t_matrix = np.vstack((t_matrix, temp))
    #EVENTS.to_csv(f'{run_spec.fname}_event_df.csv')
    
    t_matrix = np.reshape(t_matrix, (len(time), len(state_list), len(state_list)))
    return t_matrix



'''
Runs the Markov model for a given run type and parameter values

Params is a df with all input values. defaults to presets
run_spec is an object that defines the type of run this will be i.e. intervention ages
'''
def run_markov_simple(run_spec, params = ps.PARAMS):
    #print('In run_markov_simple..')
    states = ps.ALL_STATES
    time = ps.time
    t_matrix = create_t_matrix(run_spec, params = params)
#    print(t_matrix)
    
    start_state = fxns.get_start_state(states, run_spec)
    # creates a DataFrame for the population in each state for each time point
    D_matrix = pd.DataFrame(start_state, columns=states)
    #set up trackers
    OC_incidence = np.zeros(ps.NUM_YEARS+1)
    EC_incidence = np.zeros(ps.NUM_YEARS+1)
    overall_survival = np.zeros(ps.NUM_YEARS+1)
    overall_survival[0] = 1
    age = np.zeros(ps.NUM_YEARS+1)
    age[0] = ps.START_AGE
    ac_mort = np.zeros(ps.NUM_YEARS+1)
    HSBO_tracker = np.zeros(ps.NUM_YEARS+1)
    hysterectomy_tracker = np.zeros(ps.NUM_YEARS+1)
    HSBO_pre_set_age = np.zeros(ps.NUM_YEARS+1)
    #makes code more readable
    names = dm.flip(states)
    state_list = ps.STATE_DICT['STATE_NAME'].values
    OC_state_list = [s for s in state_list if 'OC' in s and 'undetected' not in s]
    EC_state_list = [s for s in state_list if 'EC' in s and 'undetected' not in s]
    
    #defining states used to calculate summary proportions
    OC_states = []
    EC_states = []
    for o in OC_state_list:
        OC_states.append(names[o])
    for e in EC_state_list:
        EC_states.append(names[e])
        
    death_states = [names[d] for d in state_list if 'death' in d]
    
    ac_states = [names[a] for a in state_list if 'AC' in a]
    new_ec_death = [0]
    new_oc_death = [0]
    # creates population distribution at time t
    for t in time:
        if t == 0:
            Distribution = start_state
            
        temp = np.transpose(t_matrix[t]) * Distribution
        Distribution = [sum(temp[i, :]) for i in range(len(states))]
        
        age[t + 1] = age[0] + t + 1
        
        HSBO_tracker[t + 1] = (Distribution[names['HSBO']] +
                               Distribution[names['init HSBO']]+ 
                               Distribution[names['AC death HSBO']]+ 
                               Distribution[names['HSBO death comps']])
        
        if age[t + 1] >= run_spec.age_survey:
            if age[t + 1] < run_spec.age_HSBO:
                HSBO_pre_set_age[t + 1] = HSBO_tracker[t + 1]
            else: 
                HSBO_pre_set_age[t + 1] = max(HSBO_pre_set_age)
        
        hysterectomy_tracker[t + 1] = (Distribution[names['hysterectomy']] + 
                                       Distribution[names['init hysterectomy']] +
                                       Distribution[names['AC death hysterectomy']]+
                                       Distribution[names['hysterectomy death comps']])
        
        death_temp = 0
        for i in death_states:
            death_temp += Distribution[i]
        
        overall_survival[t + 1] = 1 - death_temp
        
        for i in OC_states:
            OC_incidence[t + 1] += Distribution[i]
            #print(i, Distribution[i])
        for i in EC_states:
            EC_incidence[t + 1] += Distribution[i]
            #print(i, Distribution[i])
        for i in ac_states:
            ac_mort[t + 1] += Distribution[i]
        
        #checks to make sure that cancer incidence doesn't increase after HSBO
        if OC_incidence[t + 1] - OC_incidence[t] > 0.00001 and age[t+1] > run_spec.age_HSBO:
            print('OC incidence greater than last')
            print(OC_incidence)
        if EC_incidence[t + 1] - EC_incidence[t] >0.00001 and age[t+1] > run_spec.age_HSBO:
            print('EC incidence greater than last')
            print(EC_incidence)
        D_matrix.loc[len(D_matrix)] = Distribution
        this_ec_death = D_matrix.loc[len(D_matrix)-1, names['EC death']]
        old_ec_death = D_matrix.loc[len(D_matrix)-2, names['EC death']]
        new_ec_death.append(this_ec_death - old_ec_death)
        
        this_oc_death = D_matrix.loc[len(D_matrix)-1, names['OC death']]
        old_oc_death = D_matrix.loc[len(D_matrix)-2, names['OC death']]
        new_oc_death.append(this_oc_death-old_oc_death)
        
    D_matrix.columns = list(ps.ALL_STATES.values())
    
    age_HSBO_temp = run_spec.age_HSBO if run_spec.age_HSBO != 80 else 'Never'
    age_surv_temp = run_spec.age_survey if run_spec.age_survey != 80 else 'Never'
    age_hysterectomy_temp = run_spec.age_hysterectomy if run_spec.age_hysterectomy != 80 else 'Never'
    age_oophorectomy_temp = run_spec.age_oophorectomy if run_spec.age_hysterectomy != 80 else 'Never'
    
    '''
    Bonus columns for tracking: important for error checking and calculating costs/utilities
    '''
    D_matrix['new EC death'] = new_ec_death
    D_matrix['new OC death'] = new_oc_death
    D_matrix['HSBO age'] = np.full(len(D_matrix), 
                                   age_HSBO_temp)
    D_matrix['age survey'] = np.full(len(D_matrix), 
                                     age_surv_temp)
    D_matrix['age hysterectomy'] = np.full(len(D_matrix), 
                                           age_hysterectomy_temp)
    D_matrix['age oophorectomy'] = np.full(len(D_matrix), 
                                           age_oophorectomy_temp)
    strategy = run_spec.label
    D_matrix['strategy'] = np.full(len(D_matrix), strategy)
    D_matrix['gene'] = np.full(len(D_matrix), run_spec.gene)
    D_matrix['oc risk level'] = np.full(len(D_matrix), 
                                        params.loc['oc lifetime risk', 'value'])
    D_matrix['ec risk level'] = np.full(len(D_matrix), 
                                        params.loc['ec lifetime risk', 'value'])
    
    D_matrix['OC incidence'] = OC_incidence
    D_matrix['EC incidence'] = EC_incidence
    D_matrix['Cancer Incidence'] = (D_matrix['OC incidence'] + 
                                    D_matrix['EC incidence'])
    
    D_matrix['Cancer Mortality'] = D_matrix['OC death'] + D_matrix['EC death']
    D_matrix['overall survival'] = overall_survival
    D_matrix['total AC death'] = ac_mort
    D_matrix['age'] = age
    D_matrix['HSBO total'] = HSBO_tracker
    D_matrix['HSBO false positive'] = HSBO_pre_set_age
    D_matrix['hysterectomy total'] = hysterectomy_tracker
    oc_risk_level = str(int(params.loc['oc lifetime risk', 'value']))
    ec_risk_level = str(int(params.loc['ec lifetime risk', 'value']))
    
    published_OC = pd.read_excel(ps.risk_data, sheet_name = run_spec.gene+'_OC')
    published_EC = pd.read_excel(ps.risk_data, sheet_name = run_spec.gene+'_EC')
    D_matrix['lifetime oc risk'] = published_OC.loc[len(published_OC)-1, 
                                                    oc_risk_level]
    D_matrix['lifetime ec risk'] = published_EC.loc[len(published_EC)-1, 
                                                    ec_risk_level]
    
    D_matrix['lifetime total risk'] = (D_matrix['lifetime oc risk'] + 
                                       D_matrix['lifetime ec risk'])
    return D_matrix, t_matrix

'''
TEST PARAMS
'''
#=============================================================================
# temp_params = ps.PARAMS.copy()
# #temp_params.loc['ec lifetime risk'] = 4
# run_spec = ps.run_type('MLH1', age_surgery = 50, age_survey = 35)
# print(run_spec.label)
# dmat, tmat = run_markov_simple(run_spec, params = temp_params)
# print("done w first test case")

# run_spec1 = ps.run_type('MLH1', age_surgery = 50, age_survey = 35)
# print(run_spec1.label)
# dmat1, tmat1 = run_markov_simple(run_spec1)
#=============================================================================
#dmat.to_csv('test_comps_2.csv')
#filenames specifies paths to D_matrices
#col_names specifies the outcome to graph

'''
************************************************************
Returns the parameter that's different from base case 
to save as part of D_matrix
************************************************************
'''   
#
def which_param_is_different(params):
    old_params = ps.PARAMS.copy()
    diff_param = 'na'
    diff_val = 'na'
    for row in params.index:
        if isinstance(old_params.loc[row, 'value'], list):
            temp_old = old_params.loc[row, 'value']
            temp_new = params.loc[row, 'value']
            if temp_old[0] != temp_new[0]:
                diff_param = row
                diff_val = str(temp_new)
                return diff_param, diff_val
        if old_params.loc[row, 'value'] != params.loc[row, 'value']:
            diff_param = row
            if 'lifetime risk' in diff_param:
                diff_val = int(params.loc[row, 'value'])
            else:
                diff_val = params.loc[row, 'value']
    return diff_param, diff_val

'''
************************************************************
Loads the strategy with greatest QALYs for each gene
************************************************************
'''
def get_bc_optimal_strategies():
    bc_results = {}
    df = pd.read_csv(ps.dump/f"{ps.F_NAME_DICT['BC_ICERS_W_DOM']}{ps.icer_version}.csv")
    
    for gene in ps.GENES:
        temp = df[df['gene']==gene].copy()
        temp.sort_values(by=[ps.QALY_COL], ascending=False, inplace=True)
        temp.reset_index(drop=True, inplace=True)
        bc_results[gene] = temp.loc[0, 'strategy']
        
    return bc_results

'''
************************************************************
Loads the strategies on the efficiency frontier for testing
************************************************************
'''
def get_bc_optim_next_best():
    bc_results = pd.read_csv(ps.dump/f"{ps.F_NAME_DICT['BC_ICERS_W_DOM']}{ps.icer_version}.csv")
    bc_results = bc_results.dropna(subset=[ps.ICER_COL])
    small_df = pd.DataFrame()
    for g in ps.GENES:
        temp = bc_results[bc_results['gene'] == g]
        temp = temp.sort_values(by = [ps.ICER_COL])
        #print(temp.tail())
        temp = temp.reset_index(drop = True)
        temp = temp.tail(2)
        small_df = small_df.append(temp)
    return small_df.reset_index(drop = True)


'''
Returns the basecase optimal strategy (or output values) based on ICERs or QALYs
If the bc_type is icers, it will return the strategies on the efficiency frontier
'''
def load_bc(val_type = 'icers', return_strategies = False,
            return_comparator = False):
    bc_results = {}
    if val_type == 'icers' or val_type == 'cost per qaly':
        df = pd.read_csv(ps.dump/f'icers_w_dominated_all_genes{ps.icer_version}.csv')
        sort_col = ps.ICER_COL
    else:
        df = pd.read_csv(ps.dump/f'base_case_all_outputs_qalys{ps.icer_version}.csv')
        sort_col = ps.QALY_COL
    
    for g in ps.GENES:
        temp = df[df['gene'] == g]
        temp = temp.sort_values(by = [sort_col], ascending = False)
        
        if val_type == 'icers':
            temp = temp.dropna(subset = [ps.ICER_COL])
        temp = temp.reset_index(drop = True)
        
        if return_strategies:
            if return_comparator:
                if len(temp) > 1:
                    bc_results[g] = [temp.loc[0, 'strategy'],
                                       temp.loc[1, 'strategy']]
                else:
                    bc_results[g] = [temp.loc[0, 'strategy']]
            else:
                bc_results[g] = temp.loc[0, 'strategy']
        else:
            if val_type == 'cost per qaly':
                bc_results[g] = temp.loc[0, ps.COST_QALY_COL]
            else:
                if len(temp) == 1 and val_type == 'icers':
                    bc_results[g] = temp.loc[0, ps.COST_QALY_COL]
                else:
                    bc_results[g] = temp.loc[0, sort_col]
    return bc_results

'''
************************************************************
Loads basecase distribution matrices to avoid re-running
the basecase when unnecessary (e.g., changes to utils/costs)
************************************************************
'''        
def load_bc_files(optimal_only = False):
    df_dict = {}
    if optimal_only:
        bc_optimal_strats = load_bc(return_strategies = True,
                                    return_comparator = True)
        
    for gene in ps.GENES:
        k = 0
        for k in range(0, len(ps.SURVEY_AGES)):
            j = 0
            if ps.SURVEY_AGES[k] == 35:
                HSBO_ages = [40, 50, 80]
            else:
                HSBO_ages = [35, 40, 50, 80]
            for j in range(0, len(HSBO_ages)):
                run_spec = ps.run_type(gene, age_surgery = HSBO_ages[j],
                                       age_survey = ps.SURVEY_AGES[k])
                if optimal_only:
                    if run_spec.label in bc_optimal_strats[gene]:
                        key = (f'{run_spec.fname}_ec_risk_0_oc_risk_'+
                               f'0{ps.sim_version}')
                        temp_df = pd.read_csv(ps.dump/f"{key}.csv")
                        df_dict[key] = temp_df
                else:
                    key = (f'{run_spec.fname}_ec_risk_0_oc_risk_'+
                           f'0{ps.sim_version}')
                    temp_df = pd.read_csv(ps.dump/f"{key}.csv")
                    df_dict[key] = temp_df
        k = 0
        #this is a separate loop since there's no surveillance + delayed ooph strategy
        #default hysterectomy age is 40
        for k in range(0, len(ps.HYSTERECTOMY_AGES)):
            run_spec = ps.run_type(gene, age_surgery = ps.HYSTERECTOMY_AGES[k],
                                   hysterectomy_alone = True)
            if optimal_only:
                if run_spec.label in bc_optimal_strats[gene]:
                    key = (f'{run_spec.fname}_ec_risk_0_oc_risk_'+
                               f'0{ps.sim_version}')
                    temp_df = pd.read_csv(ps.dump/f"{key}.csv")
                    df_dict[key] = temp_df
            else:
                key = (f'{run_spec.fname}_ec_risk_0_oc_risk_'+
                           f'0{ps.sim_version}')
                temp_df = pd.read_csv(ps.dump/f"{key}.csv")
                df_dict[key] = temp_df
            
    return df_dict
'''
************************************************************
Iterates all genes and strategies unless load_bc is in kwargs
If load_bc is in kwargs, function skips the model run
If run_tracker is in kwargs, assume the run is a PSA
************************************************************
'''       
#possible kwargs: PSA, save_folder, save_files, run_tracker, thresh
def iterate_strategies(survey_ages = ps.SURVEY_AGES, HSBO_ages = ps.HSBO_AGES,
                           hysterectomy_ages = ps.HYSTERECTOMY_AGES, save_files = True,
                           save_w_id = True,
                           params = ps.PARAMS, genes = ps.GENES, **kwargs):
    
    if 'load_bc' in kwargs:
        load_bc = kwargs.get('load_bc')
        if load_bc:
            df_container = load_bc_files()
            return df_container
        
    #df container holds all the distribution matrices produced by this run
    df_container = {}
    if 'run_tracker' in kwargs:
        run_tracker = kwargs.get('run_tracker')
    this_id = np.random.randint(0, high = 1000000)

    oc_risk_lev = str(params.loc['oc lifetime risk', 'value'])
    ec_risk_lev = str(params.loc['ec lifetime risk', 'value'])
    if 'risk_levels' in kwargs:
        print('oc risk level: ', oc_risk_lev)
        print('ec risk level: ', ec_risk_lev)
        
    changed_param, new_value = which_param_is_different(params)
    for gene in genes:
        k = 0
        #runs through all surveillance ages: 30, 35, and 80 (no surveillance)
        for k in range(0, len(survey_ages)):
            j = 0
            #runs through all HSBO ages: 35, 40, 50, and 80 (no HSBO)
            #when k == max and j == max, it runs natural history
            if survey_ages[k] == 35:
                HSBO_ages = [40, 50, 80]
            else:
                HSBO_ages = [35, 40, 50, 80]
            for j in range(0, len(HSBO_ages)):
                
                run_spec = ps.run_type(gene, age_surgery = HSBO_ages[j], 
                                       age_survey = survey_ages[k])

                d_mat, t_mat = run_markov_simple(run_spec, params)
                #if this is a PSA, don't save anything here
                if 'PSA' in kwargs:
                    d_mat['run_tracker'] = run_tracker
                    for val in params.index:
                        if 'stage dist' in val:
                            param_val = str(params.at[val, 'value'])
                        else:
                            param_val = params.loc[val, 'value']
                        d_mat[val] = param_val

                    key = (f'{run_spec.fname}_PSA_sample_{this_id}{ps.sim_version}')
                    d_mat['key'] = key
                #If this is not a PSA, then see if the distribution matrix should be saved
                else:
                    d_mat['changed param'] = changed_param
                    d_mat['param value'] = new_value
                    if save_w_id:
                        key = (f'{run_spec.fname}_ec_risk_{ec_risk_lev}_oc_risk_'+
                               f'{oc_risk_lev}{ps.sim_version}{this_id}')
                    else:
                        key = (f'{run_spec.fname}_ec_risk_{ec_risk_lev}_oc_risk_'+
                               f'{oc_risk_lev}{ps.sim_version}')
                    d_mat['fname'] = key
                    if save_files:
                        if 'save_folder' in kwargs:
                            d_mat.to_csv(ps.dump/kwargs.get('save_folder')/f'{key}.csv', 
                                index=False)
                        else:
                            d_mat.to_csv(ps.dump/f'{key}.csv', index=False)
                
                if key not in df_container.keys():
                    df_container[key] = d_mat
                else:
                    print('repeat key: ',key)
        k = 0
        #this is a separate loop since there's no surveillance + delayed ooph strategy
        #default hysterectomy age is 40
        for k in range(0, len(hysterectomy_ages)):
            run_spec = ps.run_type(gene, age_surgery = hysterectomy_ages[k],
                                   hysterectomy_alone = True)
            
            d_mat, t_mat = run_markov_simple(run_spec, params)
            if 'PSA' in kwargs:
                d_mat['run_tracker'] = run_tracker
                for val in params.index:
                    if 'stage dist' in val:
                        param_val = str(params.at[val, 'value'])
                    else:
                        param_val = params.loc[val, 'value']
                    d_mat[val] = param_val

                key = (f'{run_spec.fname}_PSA_sample_{this_id}{ps.sim_version}')
                d_mat['key'] = key

            else:
                d_mat['changed param'] = changed_param
                d_mat['param value'] = new_value
                
                if save_w_id:
                    key = (f'{run_spec.fname}_ec_risk_{ec_risk_lev}_oc_risk_'+
                               f'{oc_risk_lev}{ps.sim_version}{this_id}')
                else:
                    key = (f'{run_spec.fname}_ec_risk_{ec_risk_lev}_oc_risk_'+
                               f'{oc_risk_lev}{ps.sim_version}')
                d_mat['fname'] = key
                if save_files:
                    if 'save_folder' in kwargs:
                        d_mat.to_csv(ps.dump/kwargs.get('save_folder')/f'{key}.csv', 
                            index=False)
                    else:
                        d_mat.to_csv(ps.dump/f'{key}.csv', index=False)

            
            if key not in df_container.keys():
                df_container[key] = d_mat

    return df_container

#dfs = iterate_strategies(save_w_id = False) 


#just iterates optimal strategies for one-way sensitivity analysis
def optimal_strats_owsa(survey_ages = [80], HSBO_ages = [80],
                        hysterectomy_ages = ps.HYSTERECTOMY_AGES, save_files = True,
                        risk_level = 0, params = ps.PARAMS,
                        utils = False):
    #checks which--if any--parameter is different and produces values to store in dataframe
    changed_param, new_value = which_param_is_different(params)
    if changed_param == 'na' and save_files == False and utils == False:
        changed_param = 'risk ac death oc surg'
        new_value = 1.0
        
    optimal_strategies = get_bc_optim_next_best()
    #print(optimal_strategies)
    #container holds all df's produced by the markov model
    #DOES NOT save (there would literally be thousands)
    #instead, returns a dictionary held in temporary storage to pass to other files for operation
    df_container = {}
    for gene in ps.GENES:
        temp_optimal = optimal_strategies[optimal_strategies['gene'] == gene]
        
        strategies_to_test = temp_optimal['strategy'].to_list()
        print(strategies_to_test)
        for s in strategies_to_test:
            hsbo_age = int(ps.STRAT_INFO_INDEX.loc[s, 'hsbo_age'])
            hyst_age = int(ps.STRAT_INFO_INDEX.loc[s, 'hysterectomy_age'])
            
            if hyst_age < 80:
                run_spec = ps.run_type(gene, age_surgery = hyst_age, hysterectomy_alone = True)
            elif hsbo_age < 80:
                run_spec = ps.run_type(gene, age_surgery = hsbo_age)
            else:
                run_spec = ps.run_type(gene)
            print(gene, run_spec.fname)
            if utils:
                key = (f'{run_spec.fname}_ec_risk_0_oc_risk_'+
                               f'0{ps.sim_version}')
                d_mat = pd.read_csv(ps.dump/f"{key}.csv")
                df_container[key] = d_mat
            else:
                d_mat, t_mat = run_markov_simple(run_spec, params)
                d_mat['changed param'] = changed_param
                d_mat['param value'] = new_value
                key = f'{run_spec.fname}_{changed_param}_{new_value}{ps.sim_version}'
                
                if key not in df_container.keys():
                    df_container[key] = d_mat
                else:
                    print('repeat key: ', key)
            
    return df_container 


#test = optimal_strats_owsa(utils=True)


