# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:18:37 2019

@author: ers2244

Presets for lynch-GYN model

"""

import pandas as pd
import numpy as np
import data_manipulation as dm
from matplotlib import rcParams
import data_io_LS as data_io

#Define Fonts for plotting
rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Gil Sans MT']
'''
************************************************************
                  DIRECTORIES/FOLDERS
************************************************************
'''
data_repo = data_io.INPUT
dump = data_io.OUTPUT
dump_psa = data_io.OUTPUT_PSA
dump_figs = data_io.OD_FIGS
#also define the version name to append to output files
sim_version = '_03_25_20_nh'
icer_version = '_03_25_20_nh'

#Filenames set naming conventions for all outputs
FILE_NAMES = pd.read_csv(data_io.F_NAMES)
F_NAME_DICT = dict(zip(FILE_NAMES['f_type'].dropna().to_list(),
                       FILE_NAMES['f_name'].dropna().to_list()))



'''
************************************************************
                  MODEL PRESETS
************************************************************
'''

#Genes for running
GENES = ['MLH1', 'MSH2', 'MSH6', 'PMS2']

#Time parameters
START_AGE = 25
CYCLE_LENGTH = 1 #1 year
NUM_YEARS = 50
END_AGE = START_AGE + NUM_YEARS
NUM_CYCLES = NUM_YEARS/CYCLE_LENGTH  # years

time = range(int(NUM_CYCLES))
age_time = range(START_AGE, END_AGE)

HSBO_AGES = [35, 40, 50, 80]
SURVEY_AGES = [30, 35, 80]
HYSTERECTOMY_AGES = [40]

d_rate = 0.03

STATE_DICT = pd.read_excel(data_io.STATE_NAMES)
STATE_DICT['STATE_NUM'] = STATE_DICT['STATE_NUM'].astype(int)

ALL_STATES = dict(zip(STATE_DICT['STATE_NUM'].to_list(),
                      STATE_DICT['STATE_NAME'].to_list()))

CONNECTIVITY = pd.read_excel(data_io.CONNECT_MATRIX)

#Determines whether or not to include no intervention as viable strategy
EXCLUDE_NH = False
#Determines whteher or not figures should be saved
#Switch to False for testing
SAVE_FIGS = True

class run_type:
    #preset for run type is natural history
    def __init__(self, gene, age_surgery = 80, age_survey = 80, 
                 hysterectomy_alone = False, risk_level = 0):
        self.gene = gene
        self.risk_level = risk_level
        self.age_survey = age_survey
        self.hysterectomy_alone = hysterectomy_alone
        if hysterectomy_alone == True:
            self.age_hysterectomy = age_surgery
            #if delaying oophorectomy, set age to 50 (natural menopause equivalent)
            self.age_oophorectomy = 50
            if self.age_hysterectomy == self.age_oophorectomy:
                print('hysterectomy and oophorectomy ages are equal, changing hyst age to 40')
                self.age_hysterectomy = 40
            self.age_HSBO = 80
            self.fname = f'{gene}_delayed_ooph_hyst_age_{self.age_hysterectomy}'
            self.label = f'Hyst+Salp: {self.age_hysterectomy}, Oophorectomy: 50'
            
        else:
            self.age_hysterectomy = 80
            self.age_oophorectomy = 80
            self.age_HSBO = age_surgery
            #if age at HSBO and at surveillance are the same, revert to HSBO alone
            if self.age_HSBO <= self.age_survey:
                self.age_survey = 80
                
            if self.age_HSBO == 80 and self.age_survey == 80 and self.hysterectomy_alone == False:
                self.label = 'Nat Hist'
                self.fname = f'{gene}_nat_hist'
            elif self.age_HSBO == 80 and self.age_survey < 80:
                self.label = 'HSBO: Never, Survey: ' + str(self.age_survey)
                self.fname = f'{gene}_hsbo_never_survey{self.age_survey}'
            elif self.age_HSBO < 80 and self.age_survey >= 80:
                self.label = 'HSBO: ' + str(self.age_HSBO) + ', Survey: Never'
                self.fname = f'{gene}_hsbo_age_{self.age_HSBO}_survey_never'
            else:
                self.label = 'HSBO: ' + str(self.age_HSBO) + ', Survey: ' + str(self.age_survey)
                self.fname = f'{gene}_hsbo_age_{self.age_HSBO}_survey_{self.age_survey}'
        
        self.min_intervention_age = min([self.age_HSBO, self.age_hysterectomy,
                                         self.age_survey])
    

OUTCOMES = ['gene', 'strategy','QALYs', 'Unadjusted Life-Years',
            'Ovarian Cancer Mortality', 'Endometrial Cancer Mortality',
            'Ovarian Cancer Incidence', 'Endometrial Cancer Incidence',
            'Cancer Mortality', 'Cancer Incidence']

RAW_OUTCOME_COLS = ['gene', 'strategy', 'total QALYs', 'total QALYs disc',
                    'total LE', 'healthy', 'HSBO total','hysterectomy total',
                    'HSBO false positive', 'total disc cost', 'total cost',
                    'OC incidence', 'OC death', 'EC incidence', 'EC death',  
                    'Cancer Incidence', 'Cancer Mortality',
                    'HSBO death comps', 'lifetime ec risk',
                    'lifetime oc risk' ,'lifetime total risk', 'changed param',
                    'param value']

QALY_COL = 'total QALYs disc'

COST_COL = 'total disc cost'

ICER_COL = 'icers'

COST_QALY_COL = 'cost per QALY'

LE_COL = 'total LE'

'''
************************************************************
                  INPUTS FROM FILES
************************************************************
'''

#Contains info for each strategy label, ages, colors for plotting
STRAT_INFO = pd.read_csv(data_io.STRATEGIES)

#Helps for plotting--just lifetime risk inputs
LT_RISKS = pd.read_csv(data_io.INPUT/'bc_lifetime_risks.csv')

#Load in risk data
#The model pulls from risk_data, but raw_risk_data is necessary for creating the risk_data spreadsheet
risk_data = data_io.CANCER_RISK_RANGE
raw_risk_data = data_io.RAW_CANCER_RISK
model_params = data_io.MODEL_PARAMS

UTIL_HELPER = pd.read_csv(data_io.BLANK_DMAT)

STRAT_INFO.dropna(inplace=True)

ALL_STRATEGIES = STRAT_INFO['strat_label'].to_list()

STRAT_INFO_INDEX = STRAT_INFO.set_index('strat_label')

STRATEGY_DICT = dict(zip(ALL_STRATEGIES, STRAT_INFO['pretty_label'].to_list()))

#For plotting PSA results
COLOR_DICT = dict(zip(STRAT_INFO['pretty_label'].to_list(),
                      STRAT_INFO['strat_colors'].to_list()))

#For plotting efficiency frontiers
MARKER_DICT = dict(zip(STRAT_INFO['pretty_label'].to_list(),
                       STRAT_INFO['marker'].to_list()))


#### Set up and read in probabilities ####
#Read in lifetable and convert probs to rates
life_table_p = pd.read_excel(model_params, sheet_name = 'lifetable', index_col = 0)
life_table_r = -(np.log(1-life_table_p))
life_table_r.rename(columns = {'p_death':'r_death'}, inplace = True)


PARAMS = pd.read_excel(model_params, sheet_name = 'params', index_col = 0)

#Reformat stage distributions so that they're lists instead of strings
PARAMS.at['oc stage dist nat hist', 
          'value'] = dm.cell_to_list(PARAMS.loc['oc stage dist nat hist', 'value'])
PARAMS.at['ec stage dist nat hist', 
          'value'] = dm.cell_to_list(PARAMS.loc['ec stage dist nat hist', 'value'])
PARAMS.at['oc stage dist intervention', 
          'value'] = dm.cell_to_list(PARAMS.loc['oc stage dist intervention', 'value'])
PARAMS.at['ec stage dist intervention', 
          'value'] = dm.cell_to_list(PARAMS.loc['ec stage dist intervention', 'value'])


PARAMS_PSA = pd.read_excel(model_params, sheet_name = 'params_PSA', index_col = 0)
PARAMS_PSA.at['oc stage dist nat hist', 
              'value'] = dm.cell_to_list(PARAMS_PSA.loc['oc stage dist nat hist', 'value'])
PARAMS_PSA.at['ec stage dist nat hist', 
              'value'] = dm.cell_to_list(PARAMS_PSA.loc['ec stage dist nat hist', 'value'])
PARAMS_PSA.at['oc stage dist intervention', 
              'value'] = dm.cell_to_list(PARAMS_PSA.loc['oc stage dist intervention', 'value'])
PARAMS_PSA.at['ec stage dist intervention', 
              'value'] = dm.cell_to_list(PARAMS_PSA.loc['ec stage dist intervention', 'value'])

#FORMATTED_PARAMS = dict(zip(PARAMS.index.to_list(), PARAMS.formatted_param.to_list()))

#### Set up and read in costs ####
raw_costs = pd.read_excel(model_params, sheet_name = 'costs')

blank_costs = pd.read_excel(model_params, sheet_name = 'blank_costs')

#### Set up and read in utilities ####
utilities = pd.read_excel(model_params, sheet_name = 'util_table_raw', 
                          index_col = 0)
raw_utils = pd.read_excel(model_params, sheet_name = 'util_table_raw', 
                          index_col = 0)

raw_utils_low = pd.read_excel(model_params, sheet_name = 'util_table_raw_low', 
                              index_col = 0)
raw_utils_up = pd.read_excel(model_params, sheet_name = 'util_table_raw_up', 
                             index_col = 0)

#Allows for easy access to the upper and utility bounds
UTIL_ENDS = {'low': raw_utils_low,
             'high': raw_utils_up}


UTIL_VARS = np.array(raw_utils.columns)
keep_arr = np.full(len(UTIL_VARS), True)
for i in range(0, len(UTIL_VARS)):
    #exclude states that won't be varied in sensitivity analysis (e.g., healthy, death)
    if (UTIL_VARS[i] not in ALL_STATES.values() or 'death' in UTIL_VARS[i] or 
        UTIL_VARS[i] == 'healthy'):
        
        keep_arr[i] = False
#defines utility values that should be changed in sensitivity analyses
UTIL_VARS = UTIL_VARS[keep_arr == True]


#helps with setting up full utility matrix in sens.py
DISUTIL_COLS = ['init HSBO', 'init hysterectomy',
                'surgical comps','gyn surveillance', 
                'undetected OC', 'undetected EC']

PARAM_FMT = pd.read_excel(model_params, sheet_name = 'owsa_dict')
FORMATTED_PARAMS = dict(zip(PARAM_FMT['param'].to_list(),
                            PARAM_FMT['formatted param'].to_list()))
PARAM_ORDER = dict(zip(PARAM_FMT['param'].to_list(),
                       PARAM_FMT['order'].to_list()))

