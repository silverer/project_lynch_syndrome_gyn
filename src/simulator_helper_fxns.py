# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:46:21 2020

@author: ers2244
"""
import pandas as pd
import numpy as np
import probability_functions_lynchGYN as pf
import presets_lynchGYN as ps

# Turns dictionaries of states and connections into a dataframe
def dict_to_connect_matrix():
    c_matrix = ps.CONNECTIVITY.to_numpy()
    
    return c_matrix

# initializes the start state
def get_start_state(states, run_spec):
    start_state = np.zeros((1, len(states)))
    if run_spec.age_HSBO == ps.START_AGE:
        start_state[0][1] = 1.
    elif run_spec.age_HSBO > ps.START_AGE:
        start_state[0][0] = 1.
    else:
        start_state[0][2] = 1.
    return start_state

#create a dataframe that's compatible with t_matrix
def get_cancer_survival(params):
    stages = ['local', 'regional', 'distant']
    oc_idx = [s + ' oc surv' for s in stages]
    
    oc_stage_states = ['OC ' + s for s in stages]
    init_oc_stage_states = ['init OC ' + s for s in stages]
    oc_stage_surv = pd.DataFrame()
    for i in range(0, len(oc_stage_states)):
        oc_stage_surv.loc[oc_stage_states[i], 
                          'five_year_OS'] = params.loc[oc_idx[i], 'value']
    i = 0
    for i in range(0, len(init_oc_stage_states)):
        oc_stage_surv.loc[init_oc_stage_states[i], 
                          'five_year_OS'] = params.loc[oc_idx[i], 'value']
    
    ec_idx = [s + ' ec surv' for s in stages]
    ec_stage_states = ['EC ' + s for s in stages]
    init_ec_stage_states = ['init EC ' + s for s in stages]
    ec_stage_surv = pd.DataFrame()
    for i in range(0, len(ec_stage_states)):
        ec_stage_surv.loc[ec_stage_states[i], 
                          'five_year_OS'] = params.loc[ec_idx[i], 'value']
    i = 0
    for i in range(0, len(init_ec_stage_states)):
        ec_stage_surv.loc[init_ec_stage_states[i], 
                          'five_year_OS'] = params.loc[ec_idx[i], 'value']
    
    
    
    return oc_stage_surv, ec_stage_surv

'''
Builds a dataframe that will be used to set the age at which events occur
Useful for detecting bugs in probabilities, since this shows which probabilities
are used at each age
'''
def build_event_df(run_spec, params):
    event_df = pd.DataFrame()
    event_df['age'] = ps.age_time
    
    temp_age_survey = run_spec.age_survey - 1
    temp_age_HSBO = run_spec.age_HSBO - 1
    
    if run_spec.hysterectomy_alone == True:
        temp_age_hysterectomy = run_spec.age_hysterectomy - 1
        temp_age_oophorectomy = run_spec.age_oophorectomy - 1
        temp_age_no_ovaries = run_spec.age_oophorectomy - 1
        temp_age_HSBO = 80
        temp_age_survey = 80
    elif run_spec.age_survey < 80:
        temp_age_hysterectomy = 80
        temp_age_oophorectomy = 80
        #since some will be eligible for Hyst-BSO w/surveillance, set the temp age
        #to the earliest possible age at Hyst-BSO
        temp_age_no_ovaries = temp_age_survey
    else:
        temp_age_hysterectomy = 80
        temp_age_oophorectomy = 80
        temp_age_no_ovaries = temp_age_HSBO
        
    nodes_oc, risk_probs_oc, risk_rates_oc = pf.cumul_prob_to_annual(ps.risk_data,
                                                                     run_spec.gene+'_OC',
                                                                     params.loc['oc lifetime risk', 
                                                                                'value'])
    nodes_ec, risk_probs_ec, risk_rates_ec = pf.cumul_prob_to_annual(ps.risk_data, 
                                                                     run_spec.gene+'_EC',
                                                                     params.loc['ec lifetime risk', 
                                                                                'value'])
   
    #Set the risk of EC and OC
    event_df['ec risk'] = event_df.apply(lambda x: pf.pw_choose_prob(x['age'], 
                                                    risk_probs_ec, nodes_ec),
                                            axis = 1)
    event_df['oc risk'] = event_df.apply(lambda x: pf.pw_choose_prob(x['age'], 
                                                    risk_probs_oc, nodes_oc),
                                            axis = 1)
    event_df['oc risk_og'] = event_df.apply(lambda x: pf.pw_choose_prob(x['age'], 
                                                    risk_probs_oc, nodes_oc),
                                            axis = 1)
    #This just converts the rate to the a probability
   
    event_df.loc[event_df['age'] > temp_age_HSBO, 'ec risk'] = 0.0
    event_df.loc[event_df['age'] > temp_age_HSBO, 'oc risk'] = 0.0
    event_df.loc[event_df['age'] > temp_age_hysterectomy, 'ec risk'] = 0.0
    #If oophorectomy + hysterectomy, then attenuate the risk of OC due to salpingectomy
    if temp_age_oophorectomy < 79:
        event_df['oc risk rate'] = event_df.apply(lambda x: pf.prob_to_rate(x['oc risk'],
                                                                        1),
                                                    axis = 1)
        
        cond = ((event_df['age'] > temp_age_hysterectomy) & (event_df['age']<= temp_age_oophorectomy))
        event_df.loc[cond,
                     'oc risk rate'] = event_df['oc risk rate'] * params.loc['risk oc tubal ligation',
                                                                              'value']
        event_df['oc risk'] = event_df.apply(lambda x: pf.rate_to_prob(x['oc risk rate'],
                                                                        1),
                                                        axis = 1)
    else:
        event_df['oc risk rate'] = np.nan
    
    event_df['oc stage dist type'] = 'oc stage dist nat hist'
    event_df['ec stage dist type'] = 'ec stage dist nat hist'
    
    #Regardless of intervention, EC will have a downstaging benefit 
    event_df.loc[event_df['age'] >= run_spec.min_intervention_age - 1,
                 'ec stage dist type'] = 'ec stage dist intervention'
    
    #if there's a set age for removing the ovaries, then apply the oc stage dist
    #for intervention to that age and older
    #doesn't apply to surveillance since no OC downstaging
    if temp_age_no_ovaries != temp_age_survey:
        
        #When everyone has set HSBO age, only apply intervention staging to the year of HSBO
        event_df.loc[event_df['age'] >= temp_age_no_ovaries,
                     'oc stage dist type'] = 'oc stage dist intervention'
        event_df.loc[event_df['age'] > temp_age_no_ovaries,
                     'oc risk'] = 0.0
    else:
        if temp_age_survey < 79 and temp_age_HSBO < 79:
            event_df.loc[event_df['age'] >= temp_age_HSBO,
                         'oc stage dist type'] = 'oc stage dist intervention'
            event_df.loc[event_df['age'] > temp_age_HSBO,
                         'oc risk'] = 0.0
    #temporary assignment/placeholder
    event_df['risk ac death oc surg'] = 1.0
    #if anyone has had oophorectomy < age 50, then apply an added risk of mortality
    #added mortality risk applies beyond age 50, the average menopause age
    if temp_age_no_ovaries < 49:
        event_df.loc[(event_df['age'] >= temp_age_no_ovaries),
                     'risk ac death oc surg'] = params.loc['risk ac death oc surg',
                                                             'value']
    #set up new columns to add to event df
    new_cols = ['true_pos_endo_surv', 'false_neg_endo_surv', 'true_neg_endo_surv',
                'false_pos_endo_surv', 'true_pos_oc_surv', 'false_neg_oc_surv',
                'false_pos_oc_surv', 'HSBO_death_risk', 'detected_oc',
                'detected_ec', 'undetected_ec', 'undetected_oc',
                'new_HSBO_ec', 'new_HSBO_oc', 'HSBO_comp_risk'] 
    for c in new_cols:
        event_df[c] = 0.
    #only fill these columns if surveillance is part of the strategy
    if run_spec.age_survey < 79:
        cond = (event_df['age'] >= temp_age_survey) & (event_df['age'] < temp_age_HSBO)
        event_df.loc[cond, 'true_pos_endo_surv'] = params.loc['sens endo surv', 'value']
        event_df.loc[cond, 'false_neg_endo_surv'] = 1.0 - params.loc['sens endo surv', 'value']
        event_df.loc[cond, 'true_neg_endo_surv'] = params.loc['spec endo surv', 'value']
        event_df.loc[cond, 'false_pos_endo_surv'] = 1.0 - params.loc['spec endo surv', 'value']
        
        event_df.loc[cond, 'true_pos_oc_surv'] = params.loc['sens oc surv', 'value']
        event_df.loc[cond, 'false_neg_oc_surv'] = 1.0 - params.loc['sens oc surv', 'value']
        event_df.loc[cond, 'true_neg_oc_surv'] = params.loc['spec oc surv', 'value']
        event_df.loc[cond, 'false_pos_oc_surv'] = 1.0 - params.loc['spec oc surv', 'value']
        
        event_df['detected_ec'] = event_df['true_pos_endo_surv'] * event_df['ec risk']
        event_df['detected_oc'] = event_df['true_pos_oc_surv'] * event_df['oc risk']
        
        event_df['undetected_ec'] = event_df['false_neg_endo_surv'] * event_df['ec risk']
        event_df['undetected_oc'] = event_df['false_neg_oc_surv'] * event_df['oc risk']
        
        event_df['new_HSBO_ec'] = event_df['false_pos_endo_surv'] * (1 - 
                                                        event_df['ec risk'])
        event_df['new_HSBO_oc'] = event_df['false_pos_oc_surv'] * (1 - 
                                                        event_df['oc risk'])
        
        #surgical mortality risk is relevant for surveillance age-temp_age_HSBO
        cond = (event_df['age'] >= temp_age_survey) & (event_df['age'] <= temp_age_HSBO)
        event_df.loc[cond, 'HSBO_death_risk'] = params.loc['surgical mortality',
                                                            'value']
        event_df.loc[cond, 'HSBO_comp_risk'] = params.loc['surgical complications',
                                                            'value']
    else:
        #if no surveillance,surgical mortality only applies to the age of surgery
        cond = ((event_df['age'] == temp_age_oophorectomy) | 
                (event_df['age'] == temp_age_hysterectomy) |
                (event_df['age'] == temp_age_HSBO))
        event_df.loc[cond, 'HSBO_death_risk'] = params.loc['surgical mortality',
                                                            'value']
        event_df.loc[cond, 'HSBO_comp_risk'] = params.loc['surgical complications',
                                                            'value']
    
    
    return event_df
