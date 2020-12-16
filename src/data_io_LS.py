# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:27:10 2019

@author: ers2244

Defines paths and directories. Initializes output dirs if they don't exist
"""

from pathlib import Path
import os
#Set working directory to file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parent_path = Path().resolve().parent
src = Path.cwd()

if os.path.isdir(Path(parent_path/"figures")) == False:
    print('creating figures directory')
    os.mkdir(Path(parent_path/"figures"))
if os.path.isdir(Path(parent_path/'model_outs'))==False:
    print('creating output directory')
    os.mkdir(Path(parent_path/'model_outs'))
if os.path.isdir(Path(parent_path/'psa_outs')) == False:
    print('creating PSA output directory')
    os.mkdir(Path(parent_path/'psa_outs'))

OD_FIGS = parent_path/"figures"

INPUT = parent_path/'data'
OUTPUT = parent_path/'model_outs'
OUTPUT_PSA = parent_path/'psa_outs'

F_NAMES = None
#Set up the file names for inputs
for (dirpath, dirnames, filenames) in os.walk(INPUT):
    for f in filenames:
        if f.startswith('model_input') and str(f[0]) != '~' and f.endswith("xls"):
            MODEL_PARAMS = INPUT/f
        if f.startswith('cancer_risk_rang') and str(f[0]) != '~' and f.endswith("xls"):
            CANCER_RISK_RANGE = INPUT/f
            #print(f)
        if f.startswith('raw_cancer') and str(f[0]) != '~' and f.endswith("xls"):
            RAW_CANCER_RISK = INPUT/f
            #print(f)
        if f.startswith('blank_dmat'):
            BLANK_DMAT = INPUT/f
        if 'strategy' in f:
            STRATEGIES = INPUT/f
        if f.startswith('named_connect') and str(f[0]) != '~' and f.endswith("xls"):
            #print('found state names')
            STATE_NAMES = INPUT/f
            print(STATE_NAMES)
        if f.startswith('connect') and str(f[0]) != '~' and f.endswith("xls"):
            CONNECT_MATRIX = INPUT/f
        if f.startswith('filenames') and str(f[0]) != '~':
            F_NAMES = INPUT/f



