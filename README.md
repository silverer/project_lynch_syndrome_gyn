
## File Structure

The model contains the following directories:

1. src: Scripts to run the base case analysis, sensitivity analyses, and format outputs. Also contains scripts to set up certain model inputs, including PSA distribution parameters and cancer risk ranges (described below). 
2. data: Model inputs and helper spreadsheets for running the model (described below)
3. model_outs: Model outputs for base-case analysis, threshold analyses, and one-way sensitivity analysis.
4. psa_outs: Model outputs (including figures) for probabilistic sensitivity analysis
5. figures: Figures generated by the model for the base case analysis, one-way sensitivity analysis, and threshold analyses 


### 1. src

* data_io_LS.py: Sets up the filepaths for the model.
* data_manipulation.py: Helper functions for reading in and setting up model inputs
* one_time_scripts.py: Sets up alpha and beta distribution parameters for sensitivity analysis, also establishes the cancer risk ranges that will be used to run the model.
* probability_functions_lynchGYN.py: Helper functions for converting probabilities to different time scales
* presets_lynchGYN.py: Defines input parameters for the model and a new class, run_spec, which is used to control the type of model run.
* lynch_gyn_simulator.py: Runs Markov models according to run types specified in lynch_gyn_ICER.py
* simulator_helper_fxns.py: Functions to get and set probabilities for the Markov model according to run type.
* lynch_gyn_ICER.py: This is the primary interface to all other scripts. A type of analysis is specified in main, and handles all simulation/aggregating. Note that analyses that use the multiprocessing module will need to be executed from the command line. You should run these analyses from the src directory. 
* lynch_gyn_sens.py: Sets up and runs one-way, threshold, and probabilistic sensitivity analyses.
* post_processing.py: Handles all formatting and plotting functions. 


### 2. data

* model_inputs.xlsx: All the model inputs (costs, utilties, and probabilities other than cancer risk) for the base case and sensitivity analyses. Note that you will need to run 'one_time_scripts.py' to generate the alpha and beta parameters in the sheets named 'params_PSA' and 'cost'.
* cost_params_temp.csv: Cost PSA input parameters generated by 'one_time_scripts.py'. These parameters should be manually copied into the 'costs' sheet in 'model_inputs.xlsx'
* psa_params_temp.csv: Probability PSA input parameters generated by 'one_time_scripts.py'. These parameters should be manually copied into the 'costs' sheet in 'model_inputs.xlsx'
* raw_cancer_risk.xlsx: Base case cancer risk inputs for the model
* cancer_risk_ranges.xlsx: Range of cancer risk inputs based on 'raw_cancer_risk.xlsx'. Generated by running 'one_time_scripts.py' Note that columns are stored and accessed as strings, even though they look like integers.
* filenames.csv: Helper spreadsheet for formatting fileneames.
* strategy_info.csv: Helper spreadsheet for changing the strategy name displayed on graphs and other formatted output.
* named_connect_matrix.xlsx: Helper spreadsheet to make health states more readable.
* connect_matrix.xlsx: Specifies the connectivity between health states.
* blank_dmat.csv: Blank distribution matrix for setting up cost and utility multiplier tables.
* bc_lifetime_risks.csv: Lists the basecase lifetime risk for each gene. Helps with formatting calibration plots.
