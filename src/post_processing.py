# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:22:20 2019

@author: ers2244
"""

import presets_lynchGYN as ps
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import regex as re
import lynch_gyn_simulator as sim
import matplotlib.lines as mlines
from matplotlib import rcParams

#Set master plot parameters
SMALL_SIZE = 14 
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

rcParams['font.sans-serif'] = "Arial"



'''
#########################################################

Functions to process PSA results

#########################################################
'''
import os
def load_psa_files(filestem):
    df_array = []
    for (dirpath, dirnames, filenames) in os.walk(ps.dump_psa):
        for f in filenames:
            if filestem in f:
                print(f)
                df_array.append(pd.read_csv(os.sep.join([dirpath, f])))
                print(len(df_array))
                
    df = pd.concat(df_array, ignore_index = True)
    print('loaded files')
    return df


'''
Creates a new ID unique to each PSA trial
'''
def create_new_id(df):
    print('generating new IDs')
    #decide where to split dataframe for runs
    #since every run has 12 strategies and 4 genes, slice every 48
    #if the number of strategies tested changes, this will update automatically
    multiplier = len(ps.ALL_STRATEGIES) * len(ps.GENES)
    df['new_id'] = np.nan      
    #set an indexer to slice each 48-row block
    indices = [i for i in range(0, len(df)) if i % multiplier==0]
    print(len(indices))
    #tracker will be the new ID
    tracker = 0
    for i in range(0, len(indices)):
        #as long as this isn't the last index slice, apply tracker to index[i] to index[i + 1]
        #eg, 0 through 47, 48 through 97 all get the same tracker
        if i < len(indices) - 1:
            #indices[i+1]-1 is to account for indices starting at 0
            #(so if there are 48 strategies, you really want 0 through 47)
            df.loc[indices[i]:indices[i+1]-1, 'new_id'] = tracker
            
        else:
            df.loc[indices[i]:, 'new_id'] = tracker
        if i % 100 == 0:
            print(i)
        
        tracker += 1
    print('done assigning IDs')
    return df
#test = create_new_id(df)
    
def get_one_run_icer(temp_df):
    static = temp_df.sort_values(by = [ps.COST_COL])
    static = static.reset_index(drop = True)
    temp_group = static.copy()
    #exclude NH here so that it's included in the dominated dataframe
    if ps.EXCLUDE_NH:
        temp_group = temp_group[temp_group['strategy'] != ps.STRATEGY_DICT['Nat Hist']]
        temp_group = temp_group.reset_index(drop = True)
        
    #temp_group = temp_group.sort_values(by = [ps.COST_COL])
    #print(static.head())
    #print(temp_group.head())
    num_rows = len(temp_group)
    row = 0
    while row < num_rows - 1:
        #Each group is already sorted by cost, so no need to do that here
        #Eliminates strategies that are more costly and less effective
        if temp_group.loc[row+1, ps.QALY_COL] < temp_group.loc[row, ps.QALY_COL]:
            temp_group = temp_group.drop([temp_group.index[row+1]])
            num_rows = len(temp_group)
            temp_group = temp_group.reset_index(drop = True)
            row = 0
        else:
            row += 1
    temp_group.loc[:, ps.ICER_COL] = 0.0
    if len(temp_group) > 1:
        num_rows = len(temp_group)
        row = 1
        while row < num_rows:
            temp_group.loc[row, 'icers'] = (
                        (temp_group.loc[row, ps.COST_COL] - temp_group.loc[row - 1, ps.COST_COL])/
                        (temp_group.loc[row, ps.QALY_COL] - temp_group.loc[row - 1, ps.QALY_COL]))
            if temp_group.loc[row, 'icers'] < temp_group.loc[row-1, 'icers']:
                temp_group = temp_group.drop(temp_group.index.values[row-1])
                temp_group = temp_group.reset_index(drop = True)
                num_rows = len(temp_group)
                row = row - 1
            else:
                row += 1
                
    icer_dict = dict(zip(temp_group['strategy'].to_list(),
                             temp_group['icers'].to_list()))
    static['icers'] = static['strategy'].map(icer_dict)
    return static
    
'''
Calculate ICERs for each PSA trial
'''
def get_icers_psa(psa_outs):
    #psa_outs = create_new_id(psa_outs)
    #INPUT: dataframe produced by generate_ce functions
    #OUTPUT: ce_table with dominated strategies eliminated and icers added
    dt = datetime.datetime.now()
    print(dt.strftime("%Y-%m-%d %H:%M"),'     getting ICERs')
    #Group the outputs to speed up processing and encapsulate each run + gene
    grouped = psa_outs.sort_values(by = ps.COST_COL).groupby(['new_id', 'gene'])
    print(len(grouped))
    
    all_results = grouped.apply(get_one_run_icer)
    all_results = all_results.reset_index(drop = True)
    
    dt = datetime.datetime.now()
    print(dt.strftime("%Y-%m-%d %H:%M"),'     done getting ICERs')
    return all_results
    
#Selects a random subsample of PSA outputs if there were extra runs
def select_subsample_psa(df, sample = 10000):
    temp = df.drop_duplicates(subset=['new_id'])
    temp = temp.sample(n = sample)
    temp = temp['new_id'].to_list()
    new = df[df['new_id'].isin(temp)]
    new = new.reset_index(drop = True)
    return new

#Takes the full outputs and drops the suboptimal strategies for each trial
def get_psa_results(full_outputs, **kwargs):
    t0 = time.time()
    #create a new id unique to each run
    if 'new_id' not in full_outputs.columns.to_list():
        full_outputs = create_new_id(full_outputs)
    if 'Unnamed: 0' in full_outputs.columns.to_list():
        full_outputs.drop(columns = ['Unnamed: 0'], inplace=True)
    
    for g in ps.GENES:
        this_gene = full_outputs[full_outputs['gene'] == g]
        temp_gene = this_gene.sort_values([ps.QALY_COL],
                                              ascending=False).groupby('new_id').head(1)
        if g == ps.GENES[0]:
            optimal_df = temp_gene.copy()
        else:
            optimal_df = pd.concat([optimal_df, temp_gene], ignore_index=True)
        
    end = time.time()
    print('time to get: ', end-t0)
    return optimal_df

def generate_bc_psa_outputs(df):
    df_new = df.copy()
    optimal_strats = sim.load_bc(return_strategies=True)
    bc_optimal_df = pd.DataFrame()
    for g in ps.GENES:
        temp_strat = ps.STRATEGY_DICT[optimal_strats[g]]
        temp = df_new[df_new['gene']== g]
        temp = temp[temp['strategy'] == temp_strat]
        bc_optimal_df = bc_optimal_df.append(temp, ignore_index = True)
        
    summary_things = ['mean', 'std','median', 'min', 'max']
    cols = [ps.QALY_COL,ps.LE_COL, ps.COST_COL,
            ps.ICER_COL, 'OC death', 'EC death',
            'OC incidence', 'EC incidence', 'Cancer Mortality',
            'Cancer Incidence']
    new_cols = ['gene', 'strategy']
    for c in cols:
        for s in summary_things:
            this_col = c + '_' + s
            new_cols.append(this_col)
    
    sample_size = bc_optimal_df['gene'].value_counts()
    sample_size = sample_size[ps.GENES[0]]        
    agg_dict = {}
    for c in cols:
        agg_dict[c] = summary_things
        if c not in bc_optimal_df.columns.to_list():
            print(f'{c} not in columns!')
        else:
            bc_optimal_df[c] = bc_optimal_df[c].astype(float)
    
    summary_df = bc_optimal_df.groupby(['gene', 'strategy']).agg(agg_dict)
    num_icers = bc_optimal_df[ps.ICER_COL].isnull().groupby(bc_optimal_df['gene']).sum().to_dict()
    
    summary_df.reset_index(inplace=True)
    summary_df.columns = new_cols
    summary_df['sample_size'] = sample_size
    summary_df['trials_not_on_EF'] = summary_df['gene'].map(num_icers)
    num_above_wtp = bc_optimal_df.groupby('gene')[ps.ICER_COL].apply(lambda x: 
                                                                     (x > 100000).sum()).to_dict()
    summary_df['trials_not_on_EF'] = summary_df['gene'].map(num_above_wtp)
    return summary_df
#psa_results = pd.read_csv(ps.dump_psa/"icers_all_genes_psa_10000_samples_12_16_20.csv")

def format_mean_sd(mean, sd, money = False):
    if money:
        mean = '${:,.2f}'.format(mean)
        sd = '${:,.2f}'.format(sd)
        return f"{mean} ({sd})"
    return f"{round(mean, 2)} ({round(sd, 2)})"

#bc_results = generate_bc_psa_outputs(psa_results)
#print(bc_results['gene'].value_counts())
def format_bc_psa_outputs(bc_df):
    new_cols = ['Average QALYs (SD)', 'Average Life-Years (SD)', 
                'Average Total Cost (SD)', 'Average ICER (SD)',
                '% of Trials on Efficiency Frontier']
    formatted_df = pd.DataFrame()
    formatted_df['Gene'] = bc_df['gene']
    formatted_df['Strategy'] = bc_df['strategy']
    formatted_df['Average QALYs (SD)'] = bc_df.apply(lambda x: format_mean_sd(x['total QALYs disc_mean'],
                                                        x['total QALYs disc_std']), axis = 1)
    formatted_df['Average Life-Years (SD)'] = bc_df.apply(lambda x: format_mean_sd(x['total LE_mean'],
                                                        x['total LE_std']), axis = 1)
    formatted_df['Average Total Cost (SD)'] = bc_df.apply(lambda x: format_mean_sd(x['total disc cost_mean'],
                                                        x['total disc cost_std'], money = True), axis = 1)
    formatted_df['Average ICER (SD)'] = bc_df.apply(lambda x: format_mean_sd(x['icers_mean'],
                                                        x['icers_std'], money = True), axis = 1)
    formatted_df['% of Trials on Efficiency Frontier'] = ((bc_df['sample_size']-bc_df['trials_not_on_EF'])/
                                                            bc_df['sample_size'])
    formatted_df['% of Trials on Efficiency Frontier'] = formatted_df['% of Trials on Efficiency Frontier'].apply(
                                                                lambda x: f"{round(x*100, 2)}%")

    return formatted_df

# bc_results = pd.read_csv(ps.dump_psa/f"psa_bc_variability{ps.icer_version}.csv")
# formatted_bc = format_bc_psa_outputs(bc_results)
# print(formatted_bc)
# formatted_bc.to_csv(ps.dump_psa/f"psa_bc_variability_formatted{ps.icer_version}.csv", index=False)
#bc_results.to_csv(ps.dump_psa/f"psa_bc_variability{ps.icer_version}.csv", index=False)
import matplotlib.ticker as mtick

def generate_wtp_threshs(df):
    all_optimal = pd.DataFrame()
    for i in range(0, len(ps.GENES)):
        temp = df[df['gene'] == ps.GENES[i]]
        wtps = [0, 25000, 50000, 75000, 100000,
                125000, 150000, 175000, 200000]
        
        for w in wtps:
            temp_1 = temp[temp['icers'] <= w]
            temp_1 = temp_1.sort_values(['icers']).groupby('new_id').tail(1)
            optimal_counts = pd.DataFrame(temp_1['strategy'].value_counts())
            optimal_counts.reset_index(inplace = True)
            optimal_counts.columns = ['strategy', "runs_optimal"]
            optimal_counts['percent_optimal'] = (optimal_counts['runs_optimal']/
                                                 sum(optimal_counts["runs_optimal"]))
            optimal_counts['wtp'] = w
            optimal_counts['gene'] = ps.GENES[i]
            all_optimal = all_optimal.append(optimal_counts, 
                                             ignore_index = True)
            
    #all_optimal.to_csv(ps.dump_psa/f'cost_effectiveness_acceptab{ps.icer_version}.csv')
    return all_optimal

#ceac = generate_wtp_threshs(psa_results)

def create_ceac_ax(gene_df, ax):
    new_df = pd.DataFrame()
    gene = gene_df['gene'].drop_duplicates().to_list()[0]
    wtp_vals = gene_df['wtp'].drop_duplicates().to_list()
    strats = gene_df['strategy'].drop_duplicates().to_list()
    for w in wtp_vals:
        temp_1 = gene_df[gene_df['wtp'] == w].copy()
        for st in strats:
            if st not in temp_1['strategy'].values:
                loc = len(temp_1)
                temp_1.loc[loc, 'strategy'] = st
                temp_1.loc[loc, 'percent_optimal'] = 0
                temp_1.loc[loc, 'wtp'] = w
                
        new_df = new_df.append(temp_1, ignore_index= True)
        #print(new_df)
        
    new_df['color'] = new_df['strategy'].map(ps.COLOR_DICT)
    
    new_df = new_df.reset_index(drop = True)
    new_df['gene'] = gene
    ax.scatter(new_df['wtp'], new_df['percent_optimal'],
               label = new_df['strategy'], c = new_df['color'].to_list())
    ax.set_xticks(wtp_vals)
    for s in strats:
        temp_1 = new_df[new_df['strategy'] == s]
        temp_1.reset_index(drop = True, inplace=True)
        ax.plot(temp_1['wtp'], temp_1['percent_optimal'],
                c = temp_1.loc[0, 'color'])
    ax.set_ylim(bottom = -1, top = 101)
    ax.set_xlim(left = 0, right = 200001)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter()) 
    ax.set_ylabel('% Iterations Cost-Effective')
    ax.set_xlabel('WTP (2019 USD)')
    labs = [str(i) for i in wtp_vals]
    #print(labs)
    ax.set_title(f"$\it{gene}$")
    ax.set_xticklabels(labels = labs, rotation = 45)
    return ax
        
def graph_wtp_threshs(ceac_df_og, together = True, show = True):
    ceac_df = ceac_df_og.copy()
    if together:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                     figsize=(10,10))
        ax_array = [ax1, ax2, ax3, ax4]
    else:
        ax_array = None
    
    ceac_df['percent_optimal'] *= 100
    if 'Nat Hist' in ceac_df['strategy'].to_list():
        ceac_df['strategy'] = ceac_df['strategy'].map(ps.STRATEGY_DICT)
        
    for i in range(0, len(ps.GENES)):
        #new_df = pd.DataFrame()
        #fig, ax = plt.subplots()
        temp = ceac_df[ceac_df['gene'] == ps.GENES[i]]
        if together:
            ax_array[i] = create_ceac_ax(temp, ax_array[i])
        else:
            fig, ax = plt.subplots(figsize = (6, 6))
            ax = create_ceac_ax(temp, ax)
            
            strats = temp['strategy'].drop_duplicates().to_list()
            legend_elements = []
            for s in strats:
                legend_elements.append(Patch(facecolor = ps.COLOR_DICT[s],
                                             label = s))
            plt.legend(handles = legend_elements, loc = 'center left',
                       bbox_to_anchor = (1, 0.5))
            plt.title(f'Cost-Effectiveness Acceptability Curve: $\it{ps.GENES[i]}$')
            if ps.SAVE_FIGS:
                plt.savefig(ps.dump_psa/f'ce_acceptability{ps.GENES[i]}.png', 
                            dpi = 300,
                            bbox_inches = 'tight')
            if show:
                plt.tight_layout()
                plt.show()
            else:
                plt.clf()
    if together:
        strats = ceac_df['strategy'].drop_duplicates().to_list()
        legend_elements = []
        for s in strats:
            legend_elements.append(Patch(facecolor = ps.COLOR_DICT[s],
                                         label = s))
        plt.legend(handles = legend_elements, loc = 'center left',
                   bbox_to_anchor = (1, 0.5))
        plt.suptitle('Cost Effectiveness Acceptability Curve by Gene', 
                     y = 1.02,
                     x = 0.45)
        plt.tight_layout()
        if ps.SAVE_FIGS:
            plt.savefig(ps.dump_psa/'ce_acceptability_all_genes.png', 
                        dpi = 300,
                        bbox_inches = 'tight')
        if show:
            plt.show()
        else:
            plt.clf()

#ceac_df = pd.read_csv(ps.dump_psa/'psa_ce_acceptability_10000_samples_12_16_20.csv')
#graph_wtp_threshs(ceac_df, together = True, show = True)


def plot_psa_results_circle_icer(df_og, show = True):
    df = df_og.copy()
    #df = create_summary_df(outputs)
    color_dict = ps.COLOR_DICT.copy()
    color_dict['All Other Strategies'] = '#F0F0F0'
    df['color'] = df['strategy'].map(color_dict)
    #pivot dataframe for plotting
    strats = []
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (8, 8))
    ax_array = [ax1, ax2, ax3, ax4]
    for i in range(0, len(ps.GENES)):
        temp = df[df['gene']==ps.GENES[i]]
        trials = len(temp['new_id'].drop_duplicates())
        #temp = temp.dropna()
        #sort ICERs (smallest to biggest) by trial ID
        #tail = 11 so that all strategies are captured
        temp = temp.dropna(subset = [ps.ICER_COL])
        temp = temp.sort_values([ps.ICER_COL]).groupby('new_id').tail(11)
        temp = temp.sort_values(['new_id', ps.ICER_COL])
        
        temp['icer_less_than_wtp'] = temp[ps.ICER_COL] < 100000
        test_fname = f"{ps.GENES[i]}_all_PSA_icers{ps.icer_version}.csv"
        temp.to_csv(ps.dump_psa/test_fname, index = False)
        
        temp = temp[temp['icers'] < 100000]
        #get strategy with highest QALYs below WTP for each trial
        temp = temp.sort_values([ps.QALY_COL], 
                                ascending = False).groupby('new_id').head(1)
        
        optimal_counts = pd.DataFrame(temp['strategy'].value_counts())
        
        optimal_counts.reset_index(inplace = True)
        optimal_counts.columns = ['strategy','runs_optimal']
        optimal_counts['percent_optimal'] = (optimal_counts['runs_optimal']/
                                              sum(optimal_counts['runs_optimal']))
        
        print(trials)
        optimal_counts.to_csv(ps.dump_psa/f'{ps.GENES[i]}_psa_outputs_{trials}_trials.csv',
                                             index=False)
        other_strats = 0
        for idx in optimal_counts.index:
            if optimal_counts.loc[idx, 'percent_optimal'] < .03:
                other_strats += optimal_counts.loc[idx, 'runs_optimal']
                optimal_counts.loc[idx, 'keep'] = False
            else:
                optimal_counts.loc[idx, 'keep'] = True
        
        optimal_counts = optimal_counts[optimal_counts['keep']==True]
        if other_strats > 0.005:
            optimal_counts.loc[len(optimal_counts), 
                               'strategy'] = 'All Other Strategies'
            optimal_counts.loc[len(optimal_counts)-1, 
                               'runs_optimal'] = other_strats 
        
        
        labels = optimal_counts['strategy'].to_list()
        sizes = optimal_counts['runs_optimal'].to_list()
        
        colors = optimal_counts['strategy'].map(color_dict).to_list()
        #colors = optimal_counts['color'].to_list()
        explode = [0.05 for i in range(0, len(sizes))]
        ax_array[i].pie(sizes, colors = colors, autopct='%1.1f%%', 
                        startangle=45, pctdistance=0.9, explode = explode,
                        labeldistance = 1.1)
        
        #draw circle
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        #fig = plt.gcf()
        ax_array[i].add_artist(centre_circle)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax_array[i].axis('equal')
        ax_array[i].set_title(f"$\it{ps.GENES[i]}$", y = .45)
        strats.extend(labels)
        
    legend_elements = []
    strats = np.unique(np.array(strats))
    for s in strats:
        legend_elements.append(Patch(facecolor=color_dict[s],
                                     label = s))
    ax_array[1].legend(handles = legend_elements, loc = 'center left',
                   bbox_to_anchor = (1, .5))
    plt.tight_layout()
    plt.suptitle('Probabilistic Sensitivity Analysis:\nResults from 10,000 Simulations', 
                 x = .35, y = 1.05)
    if ps.SAVE_FIGS:
        plt.savefig(ps.dump_psa/'psa_results_pies_icers.png',
                    bbox_inches = 'tight', dpi = 300)
    if show:
        plt.show()
    else:
        plt.clf()

#df = pd.read_csv(ps.dump_psa/f"icers_all_genes_psa_03_13_20.csv")
#generate_bc_psa_outputs(df)

#Handles all plotting and post-processing for PSA outputs
#Note: can pass a post-processed version with IDs and ICERs calculated to speed processing
def process_psa_results(full_outputs_og, result_type = 'icer', 
                        show_plot = True, target_sample_size = 10000):
    full_outputs = full_outputs_og.copy()
    
    dt = datetime.datetime.now()
    print(dt.strftime("%Y-%m-%d %H:%M"), '      processing outputs')
    current_strats = full_outputs.strategy.drop_duplicates().to_list()
    for i in current_strats:
        if 'HSBO' in i or 'Nat Hist' in i or 'Hyst+Salp' in i:
            full_outputs['strategy'] = full_outputs['strategy'].map(ps.STRATEGY_DICT)
            print('mapping strats')
            break
    #Checks if IDs have not been assigned yet   
    if 'new_id' not in full_outputs.columns:
        print('creating new ids')
        full_outputs = create_new_id(full_outputs)
        samples = len(full_outputs['new_id'].drop_duplicates())
        id_fname = f"{ps.F_NAME_DICT['PSA_ID']}_{samples}_samples{ps.icer_version}.csv"
        print(id_fname)
        full_outputs.to_csv(ps.dump_psa/id_fname,
                            index = False)
    else:
        print('IDs already assigned')
        
    if ps.EXCLUDE_NH:
        print('EXCLUDING NAT HIST')
        full_outputs['exclude_nat_hist'] = True
        #NH exclusion happens in the relevant processing functions to avoid data loss
    #Save a version of the dataframe with 10,000 random samples
    if len(full_outputs['new_id'].drop_duplicates()) > target_sample_size:
        print('selecting subsample')
        full_outputs = select_subsample_psa(full_outputs, 
                                            sample = target_sample_size)
        sub_fname = (f"{ps.F_NAME_DICT['PSA_ID_SUB']}_{target_sample_size}_"+
                        f"samples{ps.icer_version}.csv")
        print(sub_fname)
        full_outputs.to_csv(ps.dump_psa/sub_fname,
                            index = False)
    sample_size = len(full_outputs['new_id'].drop_duplicates())
        
    if result_type == 'icer':
        
        if ps.ICER_COL not in full_outputs.columns:
            print('getting ICER')
            #Calculate ICERs for each gene and Monte Carlo trial
            icers = get_icers_psa(full_outputs)
            actual_samps = len(icers['new_id'].drop_duplicates())
            print(actual_samps)
            icer_fname = (f"{ps.F_NAME_DICT['PSA_ICERS_ALL_GENES']}_"+
                            f"{sample_size}_samples{ps.icer_version}.csv")
            print(icer_fname)
            #Save icer df
            icers.to_csv(ps.dump_psa/icer_fname, index = False)
        else:
            icers = full_outputs.copy()
            
        plot_psa_results_circle_icer(icers, show = show_plot)
        ceac = generate_wtp_threshs(icers)
        ceac_fname = (f"{ps.F_NAME_DICT['PSA_CE_ACC']}_{sample_size}_"+
                         f"samples{ps.icer_version}.csv")
        print(ceac_fname)
        ceac.to_csv(ps.dump_psa/ceac_fname)
        graph_wtp_threshs(ceac, show = show_plot)
        dt = datetime.datetime.now()
        print(dt.strftime("%Y-%m-%d %H:%M"), 
              '    finished getting ICERs')   
        return icers
    elif result_type == 'qalys':
        optimal_df = get_psa_results(full_outputs)
        fname = f"{ps.F_NAME_DICT['PSA_OPTIM_QALYs']}_{sample_size}{ps.icer_version}.csv"
        optimal_df.to_csv(ps.dump_psa/fname)
        dt = datetime.datetime.now()
        print(dt.strftime("%Y-%m-%d %H:%M"), 
              '    finished getting optimal strategies')   
        return optimal_df
    else:
        bc_df = generate_bc_psa_outputs(full_outputs)
        fname = f"{ps.F_NAME_DICT['PSA_BC']}_{sample_size}{ps.icer_version}.csv"
        bc_df.to_csv(ps.dump_psa/fname)
        dt = datetime.datetime.now()
        print(dt.strftime("%Y-%m-%d %H:%M"), 
              '    finished getting BC optimal strategies')   
        return bc_df




#df = pd.read_csv(ps.dump_psa/'icers_all_genes_psa_10000_samples_03_25_20_nh.csv')

#process_psa_results(df, result_type = 'basecase')


'''
Functions to process OWSA outputs
'''

#takes all OWSA outputs and gets the max and min QALYs
def process_owsa_results(owsa_outputs):
    cols_to_keep = ['gene', 'param value', 'changed param','strategy', 
                    'pretty_strategy', ps.QALY_COL, ps.COST_COL, 
                    ps.COST_QALY_COL, ps.LE_COL, 'formatted_param']
    
    owsa_outputs = owsa_outputs[cols_to_keep]
        
    owsa_outputs[ps.ICER_COL] = np.nan
    owsa_outputs = owsa_outputs.reset_index(drop = True)
    bc_strats = sim.load_bc(return_strategies = True,
                            return_comparator = True)
    #grouped = owsa_outputs.groupby(['gene', 'formatted_param', 'param value'])
    output_df = pd.DataFrame()
    unique_strat_tracker = {}
    for g in ps.GENES:
        temp = owsa_outputs[owsa_outputs['gene'] == g]
        #temp = temp.sort_values(by = [ps.COST_COL])
        temp = temp.reset_index(drop = True)
        #Determine whether there's only one strategy on the efficiency frontier
        unique_strats = len(temp['strategy'].drop_duplicates().to_list())
        unique_strat_tracker[g] = unique_strats
        
        
        if unique_strats > 1:
            i = 0
            while i < len(temp) - 1:
                #print(temp.loc[i, 'strategy'])
                #print(temp.loc[i+1, 'strategy'])
                #Check if comparator is the row before or after i
                if temp.loc[i + 1, 'strategy'] == bc_strats[g][0]:
                    this_icer = ((temp.loc[i+1, ps.COST_COL] - temp.loc[i, ps.COST_COL])/
                                 (temp.loc[i+1, ps.QALY_COL] - temp.loc[i, ps.QALY_COL]))
                    this_comparator = temp.loc[i, 'strategy']
                    temp.loc[i+1, ps.ICER_COL] = this_icer
                    temp.loc[i+1, 'comparator'] = this_comparator
                    temp.loc[i+1, 'pretty_comparator'] = ps.STRATEGY_DICT[this_comparator]
                else:
                    this_icer = ((temp.loc[i, ps.COST_COL] - temp.loc[i+1, ps.COST_COL])/
                                 (temp.loc[i, ps.QALY_COL] - temp.loc[i+1, ps.QALY_COL]))
                    this_comparator = temp.loc[i+1, 'strategy']
                    temp.loc[i, ps.ICER_COL] = this_icer
                    temp.loc[i, 'comparator'] = this_comparator
                    temp.loc[i, 'pretty_comparator'] = ps.STRATEGY_DICT[this_comparator]
                i += 2
            output_df = output_df.append(temp, ignore_index = True)
        #if only one strategy on efficiency frontier, set icer to 0.0
        else:
            temp[ps.ICER_COL] = 0.0
            output_df = output_df.append(temp, ignore_index = True)
            
            
    temp = output_df.dropna(subset=[ps.ICER_COL])
    min_vals = temp.sort_values(['param value'], 
                                ascending = False).groupby(['gene', 
                                                            'changed param']).tail(1)
    max_vals = temp.sort_values(['param value'], 
                                ascending = False).groupby(['gene', 
                                                            'changed param']).head(1)
    #outputs for minimum and maximum input parameters    
    merged = pd.merge(min_vals, max_vals, on = ['gene', 'changed param'],
                      suffixes = ('_min', '_max'))
    bc_icers = sim.load_bc()
    merged['bc_icer'] = merged['gene'].map(bc_icers)
    
    merged['icer_diff'] = abs(merged[ps.ICER_COL+'_max'] - 
                              merged[ps.ICER_COL+'_min'])
    
    merged['width_upper'] = merged[ps.ICER_COL+'_max'] - merged['bc_icer']
    merged['width_lower'] = merged['bc_icer'] - merged[ps.ICER_COL+'_min']
    
    merged['order'] = merged['changed param'].map(ps.PARAM_ORDER)
    
    merged = merged.sort_values(['order']).groupby(['gene']).head(len(ps.PARAM_ORDER))
    
    #Clean up columns so that duplicate cols are minimized
    single_cols = ['strategy', 'pretty_strategy', 
                   'formatted_param',
                   'comparator', 'pretty_comparator']
    
    drop_cols = [s+'_min' for s in single_cols]
    merged = merged.drop(columns = drop_cols)
    rename_cols = [s + '_max' for s in single_cols]
    
    merged = merged.rename(columns = dict(zip(rename_cols, single_cols)))
    
    
    merged.to_csv(ps.dump/f"{ps.F_NAME_DICT['OWSA_ICERS']}{ps.icer_version}.csv")
    
    for g in ps.GENES:
        temp_merge = merged[merged['gene'] == g]
        temp_merge = temp_merge.sort_values(by = ['order'])
        fname = f"{g}_{ps.F_NAME_DICT['OWSA_ICERS']}{ps.icer_version}.csv"
        temp_merge.to_csv(ps.dump/fname, index = False)
    
    return merged


#temp_df = pd.read_csv(ps.dump/'owsa_raw_outputs_03_25_20_nh_all_strats.csv')
#test = process_owsa_results(temp_df)


def create_tornado_diagrams(owsa_out):
    
    bc_outcome = 'bc_icer'
    bc_icers = sim.load_bc()
    bc_strats = sim.load_bc(return_strategies = True, return_comparator=True)
    if type(bc_icers) == str:
        print('Error: could not load base case qalys, exiting...')
        return
    
    k = 0
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (14, 18))
    ax_array = [ax1, ax2, ax3, ax4]
    for gene in ps.GENES:
        
        new_df = owsa_out[owsa_out['gene'] == gene]
        new_df = new_df.drop_duplicates(subset=['changed param'])
        new_df = new_df.reset_index(drop = True)
        these_bc_qalys = bc_icers[gene]
        
        if len(bc_strats[gene]) <= 1:
            print(gene)
            max_outcome = ps.COST_QALY_COL + '_max'
            min_outcome = ps.COST_QALY_COL + '_min'
        else:
            max_outcome = ps.ICER_COL + '_max'
            min_outcome = ps.ICER_COL + '_min'
            
        new_df[bc_outcome] = these_bc_qalys
        
        
    # Create list of widths of bars
        width_total = []
        width_lower = []
        width_upper = []
        for i in range(len(new_df)):
            width_total.append(abs(new_df[max_outcome][i] - new_df[min_outcome][i]))
            width_lower.append(these_bc_qalys - new_df[min_outcome][i])
            width_upper.append(new_df[max_outcome][i] - these_bc_qalys)
        
        # Add to owsa_out dataframe
        new_df['total width'] = width_total
        new_df['lower width'] = width_lower
        new_df['upper width'] = width_upper
        
        new_df['abs total width'] = abs(new_df['total width'])
        
        new_df = new_df.sort_values(by=['abs total width'], ascending = False)
        new_df = new_df.reset_index(drop = True)
        new_df = new_df.loc[0:4, :]
        colors = ['#64A4A4', '#162E2E']
        
        lower_values = mpatches.Patch(color=colors[0], 
                                      label='Variable lower limit')
        upper_values = mpatches.Patch(color=colors[1], 
                                      label='Variable upper limit')
        new_df = new_df.sort_values(by = ['abs total width'])
        new_df = new_df.reset_index(drop = True)
        
        #If there's negative ICERs other than PMS2, the test strategy 
        #is no longer cost-effective
        if gene != 'PMS2':
            #If any ICERs are negative, don't plot the bars
            condition_lower = new_df[min_outcome] < 0
            condition_upper = new_df[max_outcome] < 0
            new_df.loc[condition_upper, max_outcome] = 0
            new_df.loc[condition_upper, 'upper width'] = 0
                    
            new_df.loc[condition_lower, min_outcome] = 0
            new_df.loc[condition_lower, 'lower width'] = 0
        else:
            condition_lower = ((new_df[min_outcome] < 0) &
                               (new_df['formatted_param'] == 'U Hysterectomy'))
            new_df.loc[condition_lower, min_outcome] = 0
            new_df.loc[condition_lower, 'lower width'] = 0
        
        # For each row in data frame
        for i in range(len(new_df)):
            #max value = highest value, min value = lowest value
            #if qalys assoc w/higher max value are lower than those associated w/min, then inverse relationship
            qaly_diff = new_df.loc[i, max_outcome] - new_df.loc[i, min_outcome]
            if qaly_diff > 0:
                face_colors = (colors[0], colors[1])
                ax_array[k].broken_barh([(new_df[min_outcome][i], 
                                          new_df['lower width'][i]),
                                (these_bc_qalys, new_df['upper width'][i])], 
                                        (i, 0.8), 
                                facecolors=face_colors, ec='white')
            else:
                ax_array[k].broken_barh([(new_df[min_outcome][i], 
                                          new_df['lower width'][i]),
                                (these_bc_qalys, new_df['upper width'][i])], 
                                        (i, 0.8), 
                                facecolors=(colors[0], colors[1]), ec='white')
            #these being negative means that the strategy is not cost-effective
            if ('U Hystere' in new_df.loc[i, 'formatted_param'] or
                'RR for OC' in new_df.loc[i, 'formatted_param'] or
                'U Early Menopause' in new_df.loc[i, 'formatted_param']):
                
                #If ICERs were negative, replace their bar with "dominated"
                if new_df.loc[i, max_outcome] == 0:
                    ax_array[k].text(new_df.loc[0, bc_outcome]+5000,i+.3, 
                                         'Dominated', c = colors[1],
                                         size = 14)
                        
                elif new_df.loc[i, min_outcome] == 0:
                    add_amnt = 5000 if gene != 'PMS2' else 500
                    ax_array[k].text(new_df.loc[0, bc_outcome]+add_amnt,i+.3, 
                                         'Dominated', c = colors[0],
                                         size = 14)
            
                
        fig.set_figheight(9)
        y_range = np.arange(0.4, len(new_df), 1)
        #ax_array[k].axvline(x = 100000, linestyle = 'dashed', c = 'grey')
        if gene == 'PMS2':
            new_df.loc[new_df['formatted_param'] == 'U Hysterectomy',
                       'formatted_param'] = 'U Hyst-BSO'
        ax_array[k].set_yticks(y_range)
        ax_array[k].set_yticklabels(new_df['formatted_param'])
        
        ax_array[k].axvline(x=these_bc_qalys, color='black')
        
        
        if len(bc_strats[gene]) <= 1:
            title = f"$\it{gene}$ ({new_df.loc[0, 'pretty_strategy']})*"
            ax_array[k].set_xlabel('Cost per QALY')
        else:
            #$\it{ps.GENES[i]}$
            title = f"$\it{gene}$ ({new_df.loc[0, 'pretty_strategy']} vs {new_df.loc[0, 'pretty_comparator']})"
            ax_array[k].set_xlabel('ICER')
        
        ax_array[k].set_title(title)
        k += 1
        
    plt.legend(handles=[lower_values, upper_values],
               bbox_to_anchor=(-1.5,-.5,1.8,0.2), loc="lower center",
               mode="expand", fontsize = 'medium', ncol=2)
    plt.suptitle('One-Way Sensitivity Analysis by Gene', x = 0.55, y = 1.02)
    plt.tight_layout()
    filename = 'tornado_diagram_all_genes'+ ps.icer_version +'_with_costs.png'
    if ps.SAVE_FIGS:
        plt.savefig(ps.dump_figs/filename, bbox_inches = 'tight', dpi = 400)
    plt.show() 

#df = pd.read_csv(ps.dump/f"{ps.F_NAME_DICT['OWSA_ICERS']}{ps.icer_version}.csv")

#create_tornado_diagrams(df)


    
#create_tornado_diagrams_QALYs(test)
# =============================================================================
# owsa = pd.read_csv(ps.dump/'process_owsa_results_icer_02_24_20.csv')
# create_tornado_diagrams(owsa)
# =============================================================================

'''
Functions to plot utility threshold and risk threshold analyses
'''

def unpack_str_list(str_list):
    new = str_list.replace('[', '')
    new = new.replace(']', '')
    new = new.replace("'", '')
    return new
    
def plot_util_thresholds_surgery(df):
    rcParams.update({'font.size': 12})
    
    df['pretty_strategy'] = df['strategy'].map(ps.STRATEGY_DICT)
    df['color'] = df['pretty_strategy'].map(ps.COLOR_DICT)
    df['new_changed_param'] = df['changed param'].apply(unpack_str_list)
    all_params = df['new_changed_param'].drop_duplicates().to_list()
    
    
    df['fmt_param'] = df['new_changed_param'].map(ps.FORMATTED_PARAMS)
    
    for g in ps.GENES:
        i = 0
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        ax_array = [ax1, ax2, ax3, ax4]
        
        df_gene = df[df['gene']==g]
        labels = df_gene['pretty_strategy'].drop_duplicates().to_list()
        colors = df_gene['color'].drop_duplicates().to_list()
        for i in range(0, len(all_params)):
            temp = df_gene[df_gene['new_changed_param']==all_params[i]]
            samples = len(temp)
            if i == 0:
                print(samples)
            ax_array[i].scatter(temp['param value'], temp[ps.QALY_COL],
                                color = temp['color'], 
                                label = temp['pretty_strategy'])
            x_lab = temp.fmt_param.drop_duplicates().to_list()[0]
            ax_array[i].set_ylabel('QALYs')
            
            ax_array[i].set_title(f'Parameter: {x_lab}', fontsize = 13)
            ax_array[i].set_xlabel(x_lab)
            
            
        legend_elements = []
        for c in range(0, len(colors)):
            legend_elements.append(Patch(facecolor=colors[c],
                                         label=labels[c]))
        plt.legend(handles=legend_elements, bbox_to_anchor=(1, 0.5), 
                   loc='center left')
        plt.suptitle(f'Utility Thresholds: {g}', y = 1.02, x = 0.43)
        plt.tight_layout()
        
        plt.savefig(ps.dump/f'util_thresholds_{g}_{samples}_samples{ps.icer_version}.png', 
                    dpi = 200,
                    bbox_inches = 'tight')
        plt.clf()
        #plt.show()

def format_percent(x):
    if type(x) == str:
        try:
            x = float(x)
        except:
            print('error, wrong data type: ', x)
            print(type(x))
            return x
    x = '{:.2%}'.format(x)
    return x

#Get the comparator and optimal strategies from threshold analyses
def get_comparator_optim_strats_oneway(df_og):
    df = df_og.reset_index(drop = True)
    
    df['strategy'] = df['strategy'].map(ps.STRATEGY_DICT)
    
    df['param value'] = df['param value'].astype(float)
    og_changed_param = df.loc[0, 'changed param']
    changed_param = ps.FORMATTED_PARAMS[og_changed_param]
    if 'oc lifetime' in og_changed_param:
        df['param value'] = df['lifetime oc risk'].astype(float)
        
    elif 'ec lifetime' in og_changed_param:
        df['param value'] = df['lifetime ec risk'].astype(float)
    df = df.loc[df["changed param"]==og_changed_param, :]
    temp_changed_param = changed_param
    strategy_genes = pd.DataFrame()
    tracker = 0
    for i in range(0, len(ps.GENES)):
        temp = df[df['gene'] == ps.GENES[i]]
        #get param vals to set x axis labels
        param_vals = temp['param value'].drop_duplicates().to_list()
        temp = temp.dropna(subset=[ps.ICER_COL])
        #Exclude ICERs > $100000 WTP
        temp = temp[temp[ps.ICER_COL] <= 100000]
        temp_params = temp.sort_values(by=[ps.ICER_COL], 
                                 ascending = False).groupby(['param value']).head(2)
        comparator_strats = []
        for p in param_vals:
            param_best = temp_params.loc[temp_params["param value"]==p,:]
            param_best = param_best.drop_duplicates(subset = ["strategy"])
            param_best = param_best.reset_index(drop = True)
            if len(param_best)>1 and param_best.loc[0, "strategy"] == param_best.loc[1, "strategy"]:
                print("err")
            if len(param_best) > 1:
                this_label = param_best.loc[0, "strategy"] + " vs. " + param_best.loc[1, "strategy"]
            else:
                this_label = param_best.loc[0, "strategy"] + " (No Comparator)"
            
            comparator_strats.append(this_label)
        comparator_strats = np.unique(comparator_strats).tolist()
        for k in range(0, len(comparator_strats)):
            strategy_genes.loc[tracker, "gene"] = ps.GENES[i]
            strategy_genes.loc[tracker, "comparator_strats"] = comparator_strats[k]
            tracker += 1
    return strategy_genes

def comparator_str_wrapper():
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(ps.dump):
        for filename in filenames:
            if filename.startswith("threshold_icers") and filename.endswith(f"{ps.icer_version}.csv"): 
                file_list.append(filename)
    all_strat_genes = pd.DataFrame()
    for f in file_list:
        df = pd.read_csv(ps.dump/f)
        if "lifetime risk" in df.loc[0, "changed param"]:
            df_ec = df.loc[df["changed param"]=="ec lifetime risk", :]
            df_ec = df_ec.reset_index(drop = True)
            res = get_comparator_optim_strats_oneway(df_ec)
            all_strat_genes = pd.concat([all_strat_genes, res], ignore_index=True)
            df_oc = df.loc[df["changed param"]=="oc lifetime risk", :]
            df_oc = df_oc.reset_index(drop = True)
            res = get_comparator_optim_strats_oneway(df_oc)
            all_strat_genes = pd.concat([all_strat_genes, res], ignore_index=True)
        else:
            res = get_comparator_optim_strats_oneway(df)
            all_strat_genes = pd.concat([all_strat_genes, res], ignore_index=True)
    all_strat_genes = all_strat_genes.drop_duplicates(subset = ["comparator_strats"])
    all_strat_genes.to_csv(ps.data_repo/"comparator_strats.csv", index = False)
#comparator_str_wrapper()

def plot_one_way_optim(df_og):
    
    df = df_og.reset_index(drop = True)
    comparator_strat_markers = pd.read_csv(ps.data_repo/"comparator_strats_markers.csv")
    comparator_strat_markers = dict(zip(comparator_strat_markers["comparator_strats"].to_list(),
                                        comparator_strat_markers["marker"].to_list()))
    df['strategy'] = df['strategy'].map(ps.STRATEGY_DICT)
    df['marker'] = df['strategy'].map(ps.MARKER_DICT)
    df['color'] = df['strategy'].map(ps.COLOR_DICT)
    df['param value'] = df['param value'].astype(float)
    og_changed_param = df.loc[0, 'changed param']
    changed_param = ps.FORMATTED_PARAMS[og_changed_param]
    if 'oc lifetime' in og_changed_param:
        df['param value'] = df['lifetime oc risk'].astype(float)
        
    elif 'ec lifetime' in og_changed_param:
        df['param value'] = df['lifetime ec risk'].astype(float)
    
    df = df.loc[df["changed param"]==og_changed_param, :]
    #set up 4-panel figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                  figsize = (11, 9))
    ax_array = [ax1, ax2, ax3, ax4]
    
    temp_changed_param = changed_param
    strategy_genes = {}
    for i in range(0, len(ps.GENES)):
        temp = df[df['gene'] == ps.GENES[i]]
        #get param vals to set x axis labels
        param_vals = temp['param value'].drop_duplicates().to_list()
        temp = temp.dropna(subset=[ps.ICER_COL])
        #Exclude ICERs > $100000 WTP
        temp = temp[temp[ps.ICER_COL] <= 100000]
        temp_params = temp.sort_values(by=[ps.ICER_COL], 
                                 ascending = False).groupby(['param value']).head(2)
        temp = temp.sort_values(by=[ps.ICER_COL], 
                                 ascending = False).groupby(['param value']).head(1)
        strats = []
        comparator_strats = []
        marker_strats = []
        comp_pretty = []
        #create unique markers for each set of strategies vs. comparators
        for p in param_vals:
            param_best = temp_params.loc[temp_params["param value"]==p,:]
            param_best = param_best.drop_duplicates(subset = ["strategy"])
            param_best = param_best.reset_index(drop = True)
            if len(param_best) > 1:
                this_label = param_best.loc[0, "strategy"] + " vs. " + param_best.loc[1, "strategy"]
                comp_pretty.append(param_best.loc[0, "strategy"] + "\nvs. " + param_best.loc[1, "strategy"])
            else:
                this_label = param_best.loc[0, "strategy"] + " (No Comparator)"
                comp_pretty.append(param_best.loc[0, "strategy"] + "\n(No Comparator)")

            temp.loc[temp["param value"]==p, "comparator_strat"] = this_label
            comparator_strats.append(this_label)
            marker_strats.append(comparator_strat_markers[this_label])
        marker_strats = np.unique(marker_strats).tolist()
        comparator_strats = np.unique(comparator_strats).tolist()
        comp_pretty = np.unique(comp_pretty).tolist()
        for k in range(0, len(comparator_strats)):
            temp_strat = temp[temp['comparator_strat'] == comparator_strats[k]]
            ax_array[i].scatter(temp_strat['param value'],
                                 temp_strat[ps.ICER_COL],
                                 marker=marker_strats[k],
                                 color = 'k', label = comparator_strats[k])
        
        ax_array[i].set_xlabel(f'Value of {temp_changed_param}')
        ax_array[i].set_ylabel('ICER')
        
        ax_array[i].set_ylim(bottom = min(temp[ps.ICER_COL])-2000,
                             top = max(temp[ps.ICER_COL])+2000)
        title = f"$\it{ps.GENES[i]}$: Optimal Strategies"
        ax_array[i].set_title(title)
        for k in range(0, len(param_vals)):
             param_vals[k] = round(param_vals[k], 3)
        legend_elements = []
        for k in range(0, len(marker_strats)):
            legend_elements.append(mlines.Line2D([], [], color='black', 
                                    marker=marker_strats[k], linestyle='None',
                                    markersize=10, label=comp_pretty[k]))
            
        ax_array[i].legend(handles = legend_elements, framealpha=0.1,edgecolor="grey") 
    plt.suptitle((f"One-Way Sensitivity Analysis:\nImpact of {temp_changed_param}"+
                  " on ICERs"),
                  y = 1.01, x = 0.5)
    plt.tight_layout()
    if ps.SAVE_FIGS:
        fname = f'owsa_{og_changed_param}_icers{ps.icer_version}.png'
        plt.savefig(ps.dump_figs/fname, bbox_inches = 'tight',
                     dpi = 300)
    #plt.show()


def plot_all_threshold_analyses():
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(ps.dump):
        for filename in filenames:
            if filename.startswith("threshold_icers") and filename.endswith(f"{ps.icer_version}.csv"): 
                file_list.append(filename)
    all_strat_genes = pd.DataFrame()
    for f in file_list:
        df = pd.read_csv(ps.dump/f)
        if "lifetime risk" in df.loc[0, "changed param"]:
            df_ec = df.loc[df["changed param"]=="ec lifetime risk", :]
            df_ec = df_ec.reset_index(drop = True)
            plot_one_way_optim(df_ec)
            df_oc = df.loc[df["changed param"]=="oc lifetime risk", :]
            df_oc = df_oc.reset_index(drop = True)
            plot_one_way_optim(df_oc)
        else:
            plot_one_way_optim(df)
#plot_all_threshold_analyses()
 
def plot_risk_thresholds_cancer_incidence(df):
    rcParams.update({'font.size': 12})
    df['param value'] = df['param value'].astype(int)
    
    df['pretty_strategy'] = df['strategy'].map(ps.STRATEGY_DICT)
    df['color'] = df['pretty_strategy'].map(ps.COLOR_DICT)
    risk_types = ['ec lifetime risk', 'oc lifetime risk']
    vals = ['lifetime ec risk', 'lifetime oc risk']
    labs = ['Lifetime EC Risk', 'Lifetime OC Risk']
    for g in ps.GENES:
        
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,8))
        i = 0
        ax_array = [ax1, ax2]
        df_gene = df[df['gene']==g]
        abs_max = max(df_gene['Cancer Incidence']) + 0.05
        abs_min = 0.0
        labels = df_gene['pretty_strategy'].drop_duplicates().to_list()
        colors = df_gene['color'].drop_duplicates().to_list()
        for i in range(0, len(risk_types)):
            temp = df_gene[df_gene['changed param']==risk_types[i]]
            temp = temp.sort_values(by = [vals[i]])
            temp['percent_risk'] = temp[vals[i]].apply(format_percent)
            ax_array[i].bar(temp['percent_risk'].astype(str), temp['Cancer Incidence'],
                            color = temp['color'])
            bottom = abs_min
            top = abs_max
            ax_array[i].set_xticklabels(labels =temp['percent_risk'].astype(str).drop_duplicates(),
                                        rotation = 90)
            rcParams.update({'font.size': 12})
            ax_array[i].set_ylim(top=top, bottom = bottom)
            ax_array[i].set_ylabel('Total Cancer Incidence')
            ax_array[i].set_xlabel(labs[i])
            ax_array[i].set_title(labs[i])
            
        k = 0
        for k in range(0, len(labels)):
            if k == 0:
                legend_elements = [Patch(facecolor=colors[k], label = labels[k])]
            else:
                legend_elements.append(Patch(facecolor=colors[k], label = labels[k]))
        plt.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, .5))
        plt.suptitle(f"Optimal Strategy and Cancer Incidence by Cancer Risk Level: {g}",
                     y = 1.02, x = 0.45)
        plt.tight_layout()
        plt.savefig(ps.dump/f'risk_thresholds_w_cancer_incidence_{g}{ps.icer_version}.png', dpi=200,
                    bbox_inches='tight')
        plt.show()
        


'''
Plot the base case outputs
'''


def create_icer_ax(icer_table_gene, ax):
    
    this_df = icer_table_gene.sort_values(by = [ps.ICER_COL])
    this_df = this_df.reset_index(drop = True)
    ax.set_ylabel('QALYs')
    ax.set_xlabel('Cost (2019 USD)')
    num_rows = len(this_df)
    for i in range(0, num_rows - 1):
        line_x = [this_df.loc[i, ps.COST_COL], this_df.loc[i+1, ps.COST_COL]]
        line_y = [this_df.loc[i, ps.QALY_COL], this_df.loc[i+1, ps.QALY_COL]]
        line_text = '${:,.2f}'.format(this_df.loc[i+1, 'icers'])
            
        line_label_x = ((line_x[1] + line_x[0])/2)
        line_label_y = ((line_y[1] + line_y[0])/2) - .02
        ax.plot(line_x, line_y, color = 'grey', linestyle = 'dashed')
        ax.text(line_label_x, line_label_y, line_text)
        
                
    y_vals = this_df[ps.QALY_COL]
    markers = this_df['marker']
    marker_labels = this_df['strategy']
    x_vals = this_df[ps.COST_COL]
    if icer_table_gene.loc[0, 'gene'] == 'PMS2':
        ax.set_xlim(left = min(x_vals)-500, right = max(x_vals)+500)
    for i in range(0, len(markers)):
        ax.scatter(x_vals[i], y_vals[i], label = marker_labels[i],
                   marker = markers[i], alpha = 1, color = 'k', s = 100)
    return ax
    

def graph_eff_frontiers(icer_table_og, together = False):
    icer_table = icer_table_og.copy()
    icer_table['strategy'] = icer_table['strategy'].map(ps.STRATEGY_DICT)
    icer_table['marker'] = icer_table['strategy'].map(ps.MARKER_DICT)
    icer_table = icer_table.dropna(subset=[ps.ICER_COL])
    icer_table = icer_table.reset_index(drop = True)
    
    if together:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                 figsize = (11, 9))
        ax_array = [ax1, ax2, ax3, ax4]
    else:
        ax_array = None
        
    
    for i in range(0, len(ps.GENES)):
        this_df = icer_table[icer_table['gene'] == ps.GENES[i]]
        this_df = this_df.reset_index(drop = True)
        if together:
            
            ax_array[i] = create_icer_ax(this_df, ax_array[i])
            ax_array[i].set_title(f"$\it{ps.GENES[i]}$")
        else:
            fig, ax = plt.subplots(figsize = (7, 5))
            ax = create_icer_ax(this_df, ax)
            plt.legend(loc = 'center left', bbox_to_anchor = (1, .5))
            plt.title(f"Efficiency Frontier: $\it{ps.GENES[i]}$")
            if ps.SAVE_FIGS:
                fname = f"{ps.GENES[i]}_eff_frontier{ps.icer_version}.eps"
                plt.savefig(fname, dpi = 300, bbox_inches = 'tight')
                
            plt.tight_layout()
            #plt.show()
    if together:
        strats = icer_table['strategy'].drop_duplicates().to_list()
        legend_elements = []
        for s in strats:
            legend_elements.append(mlines.Line2D([], [], color='black', 
                                    marker=ps.MARKER_DICT[s], linestyle='None',
                                    markersize=10, label=s))
        plt.legend(handles = legend_elements, loc = 'center left',
                    bbox_to_anchor = (1.0, 0.5))
        plt.suptitle('Efficiency Frontiers by Gene', y = 1.01, x = 0.4)
        plt.tight_layout()
        if ps.SAVE_FIGS:
            plt.savefig(ps.dump_figs/f'eff_frontiers_all_genes{ps.icer_version}.eps',
                        dpi = 400, bbox_inches = 'tight')
        
        #plt.show()
        

def create_bc_ax_qalys_cancer(gene_df_og, ax, sub_ax):
    gene_df = gene_df_og.copy()
    df = gene_df.sort_values(by = ps.QALY_COL, ascending=False)
    df = df.reset_index(drop = True)
    
    x = df['strategy'].to_list()
    y1 = df[ps.QALY_COL].to_list()
    
    y2 = df['Cancer Incidence'].to_list()
    #set the primary axis (QALYs)
    ax.plot(x, y1, marker = 'o', color = 'b',
            label = 'QALYs')
    
    ax.set_ylabel('QALYs')
    ax.yaxis.label.set_color('b')
    ax.tick_params(axis='y', colors='b')
    ax.tick_params(axis ='x', rotation = 90)
    
    #Build graph with second y-axis 
    sub_ax.set_ylabel('Cancer Incidence')
    #sub_ax.yaxis.label.set_color('firebrick')
    if gene_df.loc[0, 'gene'] != 'PMS2':
        sub_ax.set_yticks(np.arange(0, 60, step = 10))
    else:
        sub_ax.set_yticks(np.arange(0, 25, step = 5))
        
    sub_ax.set_ylim(bottom=0, top = max(y2)+5)
    
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    sub_ax.yaxis.set_major_formatter(yticks)
    #sub_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    sub_ax.tick_params(axis='y', colors='firebrick')
    sub_ax.plot(x, y2, marker = 's', color = 'firebrick',
                label = 'Cancer Incidence')
    sub_ax.yaxis.label.set_color('firebrick')
    return ax, sub_ax

def create_bc_ax_qalys_cancer_bw(gene_df_og, ax, sub_ax):
    gene_df = gene_df_og.copy()
    df = gene_df.sort_values(by = ps.QALY_COL, ascending=False)
    df = df.reset_index(drop = True)
    
    x = df['strategy'].to_list()
    y1 = df[ps.QALY_COL].to_list()
    
    y2 = df['Cancer Incidence'].to_list()
    #set the primary axis (QALYs)
    ax.plot(x, y1, marker = 'o', color = 'k',
            label = 'QALYs')
    
    ax.set_ylabel('QALYs')
    ax.yaxis.label.set_color('k')
    ax.tick_params(axis='y', colors='k')
    ax.tick_params(axis ='x', rotation = 90)
    
    #Build graph with second y-axis 
    sub_ax.set_ylabel('Cancer Incidence')
    #sub_ax.yaxis.label.set_color('firebrick')
    if gene_df.loc[0, 'gene'] != 'PMS2':
        sub_ax.set_yticks(np.arange(0, 60, step = 10))
    else:
        sub_ax.set_yticks(np.arange(0, 25, step = 5))
        
    sub_ax.set_ylim(bottom=0, top = max(y2)+5)
    
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    sub_ax.yaxis.set_major_formatter(yticks)
    #sub_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    sub_ax.tick_params(axis='y', colors='k')
    sub_ax.plot(x, y2, marker = 's',linestyle = 'dashed',
                color = 'k',
                label = 'Cancer Incidence')
    sub_ax.yaxis.label.set_color('k')
    return ax, sub_ax

from matplotlib.lines import Line2D

def plot_basecase(output_df_og, together = False, column = ps.QALY_COL,
                  select_strats = True, bw = True):
    
    output_df = output_df_og.copy()
    if ps.EXCLUDE_NH:
        output_df = output_df[output_df['strategy'] != 'Nat Hist']
        fname = f'qalys_cancer_incidence{ps.icer_version}.eps'
    else:
        if bw:
            fname = f'qalys_cancer_incidence{ps.icer_version}_with_nh_bw.eps'
        else:
            fname = f'qalys_cancer_incidence{ps.icer_version}_with_nh.eps'
    
    output_df['strategy'] = output_df['strategy'].map(ps.STRATEGY_DICT)
    
    if select_strats:
        cond = ((output_df['strategy'].str.contains('Survey Alone'))|
                (output_df['strategy'].str.contains('Survey: 30')))
        output_df = output_df[cond == False]
    
    output_df['Cancer Incidence'] *= 100
    output_df['Cancer Mortality'] *= 100
    
    if together:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))
        ax_array = [ax1, ax2, ax3, ax4]
        sub_ax_array = [a.twinx() for a in ax_array]
    else:
        ax_array = None
        sub_ax_array = None
    
    for i in range(0, len(ps.GENES)):
        temp = output_df[output_df['gene'] == ps.GENES[i]]
        temp = temp.reset_index(drop = True)
        if together:
            if bw:
                ax_array[i], sub_ax_array[i] = create_bc_ax_qalys_cancer_bw(temp, ax_array[i],
                                                                            sub_ax_array[i])
            else:
                ax_array[i], sub_ax_array[i] = create_bc_ax_qalys_cancer(temp, ax_array[i],
                                                                            sub_ax_array[i])
            ax_array[i].set_title(f"$\it{ps.GENES[i]}$")
            if bw:
                if i == 1:
                    custom_legend = [Line2D([0], [0], color = 'k',
                                            marker = 'o'),
                                     Line2D([0], [0], color = 'k',
                                            linestyle = 'dashed', marker = 's')]
                    ax_array[i].legend(custom_legend, ['QALYs', 'Cancer\nIncidence'],
                                       bbox_to_anchor = (1.3,.7))
        else:
            fig, ax = plt.subplots(figsize = (7, 7))
            sub_ax = ax.twinx()
            ax, sub_ax = create_bc_ax_qalys_cancer(temp, ax, sub_ax)
            plt.title(f"QALYs and Cancer Incidence by Strategy: $\it{ps.GENES[i]}$")
            if ps.SAVE_FIGS:
                fname = f'qalys_cancer_incidence_{ps.GENES[i]}{ps.icer_version}.eps'
                plt.savefig(ps.dump_figs/fname, dpi = 300, 
                            bbox_inches = 'tight')
                
            plt.tight_layout()
            #plt.show()
            
    if together:
        plt.suptitle(f"QALYs and Cancer Incidence by Strategy", y = 1.03)
        
        plt.tight_layout()
        if ps.SAVE_FIGS:
            plt.savefig(ps.dump_figs/fname, bbox_inches = 'tight', dpi = 300)
        #plt.show()
        



def build_outputs_ax(df_dict, ax, gene, col):
    temp = df_dict.copy()
    x_vals = ps.age_time
    for key in df_dict:
        if 'hist' in key or gene not in key:
            del temp[key]
    
#    ax.set_ylabel(col)
    ax.set_xlabel('Age')
    for key in temp:
        this_df = temp[key]
            
        y_vals = this_df.loc[1:, col] * 100
        
        this_label = ps.STRATEGY_DICT[this_df.loc[0, 'strategy']]
        split_loc = re.search("\d\d[,] S", this_label)
        if split_loc:
            this_label = re.sub(", ",",\n", this_label)
        elif this_label == 'Two-Stage Approach':
            this_label = "Two-Stage\nApproach"
        if this_df.loc[0, 'age hysterectomy'] != 'Never':
            line = '-.'
        elif this_df.loc[0, 'HSBO age'] == 'Never':
            line = '--'
        elif this_df.loc[0, 'age survey'] == 'Never':
            line = '-'
        else:
            line = ':'
        
        ax.plot(x_vals, y_vals, label = this_label, linestyle = line)
        
    ax.set_xticks([30, 40, 50, 60, 70])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        
    ax.set_title(f"$\it{gene}$")
    return ax
    


def plot_outputs_four_panel(df_dict = 'none', outcome_col = 'Cancer Incidence', 
                            save = True):
    if type(df_dict) == str:
        df_dict = sim.load_bc_files()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (11, 9))
    
    ax_array = [ax1, ax2, ax3, ax4]
    i = 0
    for i in range(0, len(ps.GENES)):
        gene = ps.GENES[i]
        ax_array[i] = build_outputs_ax(df_dict, ax_array[i], gene,
                                       outcome_col)
        
    plt.legend(loc = 'center left', bbox_to_anchor = (1, .5))
    fig.suptitle(f'{outcome_col} by Strategy and Gene', x = 0.4, y = 1.015)
    plt.tight_layout()
    if ps.SAVE_FIGS:
        png_name = f'all_genes_{outcome_col}{ps.sim_version}.png'
        
        plt.savefig(ps.dump_figs/png_name, dpi = 200, bbox_inches = 'tight')
    #plt.show()



def plot_cancer_inc_mort(df_dict = 'none'):
    if type(df_dict) == str:
        df_dict = sim.load_bc_files()
    fig, ((ax1, ax2), (ax3, ax4),
          (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize = (11, 15))
    
    ax_array_1 = [ax1, ax2, ax3, ax4]
    ax_array_2 = [ax5, ax6, ax7, ax8]
    i = 0
    
    for i in range(0, len(ps.GENES)):
        gene = ps.GENES[i]
        ax_array_1[i] = build_outputs_ax(df_dict, ax_array_1[i],
                                         gene, 'Cancer Incidence')
        ax_array_2[i] = build_outputs_ax(df_dict, ax_array_2[i],
                                         gene, 'Cancer Mortality')
        
    ax_array_1[0].text(0,1.2, 'A. Cancer Incidence', transform = ax_array_1[0].transAxes,
                       va = 'top', ha = 'center', fontsize = 18,
                       fontweight = 'bold')
    
    ax_array_2[0].text(0,1.2, 'B. Cancer Mortality', transform = ax_array_2[0].transAxes,
                       va = 'top', ha = 'center',
                       fontsize = 18,
                       fontweight = 'bold')
    
    plt.legend(bbox_to_anchor = (1, 2), loc = 'center left',
               fontsize = 'medium')
    plt.subplots_adjust(wspace=.4, hspace=.4)
    if ps.SAVE_FIGS:
        plt.savefig(ps.dump_figs/f'cancer_inc_mort{ps.icer_version}.png',
                    dpi = 300, bbox_inches = 'tight')
    #plt.show()


#plot_cancer_inc_mort()

def plot_nat_hist_outputs(df_container = 'none', outcome_cols = ['OC incidence',
                                                                 'EC incidence'], 
                                             genes = ps.GENES,):
    if type(df_container) == str:
        df_container = sim.load_bc_files()
    x_vals = ps.age_time
    gene_color_dict = {'MLH1': 'y',
                       'MSH2': 'm',
                       'MSH6': 'c',
                       'PMS2': 'b'}
    for col in outcome_cols:
        ax = plt.subplot()
        plt.xlabel('Age')
        
        plt.ylabel(col)
        title = f'{col} by Gene, Natural History'
        plt.title(title)
        dfs = dict(filter(lambda elem: 'nat_hist' in elem[0], df_container.items()))
        
        for f in dfs.keys():
            this_df = dfs[f].copy()
            this_df[col] *= 100
            y_vals = this_df.loc[1:, col]
            this_label = this_df.loc[0, 'gene']
            if 'incidence' in col:
                if 'EC' in col:
                    cancer_type = 'EC'
                    
                elif 'OC' in col:
                    cancer_type = 'OC'
                sheet = this_label + cancer_type
                targets = pd.read_excel(ps.raw_risk_data, sheet_name = sheet)
                target_ages = targets['age'].values
                colname = f"reported_risk_{cancer_type}"
                target_percents = targets[colname].values * 100
                new_label = f'Target incidence: $\it{this_label}$'
                ax.plot(target_ages, target_percents, label=new_label,
                        color = gene_color_dict[this_label], linestyle = 'dashed')
                ax.set_xlim(left=25, right=75)
            
            ax.plot(x_vals, y_vals, label = f"Modeled incidence: $\it{this_label}$", 
                    color = gene_color_dict[this_label])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        png_name = f'{title}{ps.sim_version}.eps'
        plt.savefig(ps.dump_figs/png_name, dpi = 200, bbox_inches = 'tight')
        #plt.show()
        ax.clear()

        
def format_decimal(x):
    return("%.2f" % x)
        

def format_money(x):
    if pd.isnull(x):
        return 'Dominated'
    return '${:,.2f}'.format(x)

def format_all_numbers(orig_df):
    df = orig_df.copy()
    for c in df.columns:
        if 'QALY' in c or 'LE' in c:
            df[c] = df[c].apply(format_decimal)
        elif 'gene' in c or 'strategy' in c or 'param' in c or 'marker' in c:
            skip = True
        elif 'cost' in c or 'icer' in c:
            df[c] = df[c].apply(format_money)
        else:
            df[c] = df[c].apply(format_percent)
            
    df['strategy'] = df['strategy'].map(ps.STRATEGY_DICT)
    df['Cancer Incidence (Mortality)'] = df['Cancer Incidence'] + ' ('+df['Cancer Mortality']+')'
    df['EC Incidence (Mortality)'] = df['EC incidence'] + ' ('+df['EC death']+')'
    df['OC Incidence (Mortality)'] = df['OC incidence'] + ' ('+df['OC death']+')'
    
    keep_cols = ['gene', 'strategy', ps.QALY_COL, 'total LE',
                 'Cancer Incidence (Mortality)', ps.ICER_COL, ps.COST_COL,  
                 'OC Incidence (Mortality)', 'EC Incidence (Mortality)']
    df = df[keep_cols]
    return df

def create_bc_output_table(orig_bc_outs = 'none', keep_strats = 'all'):
    keep_cols = ['gene', 'strategy', ps.QALY_COL, ps.COST_COL,
                 ps.ICER_COL,
                 'Cancer Incidence', 'Cancer Mortality']
    
    if type(orig_bc_outs) == str:
        try:
            bc_outs = pd.read_csv(ps.dump/f'base_case_all_outputs_qalys{ps.icer_version}.csv')
        except:
            'base case QALY outputs not found!'
            return 'error'
    else:
        bc_outs = orig_bc_outs.copy()
        
    fmt_df = pd.DataFrame()
    
    bc_outs['strategy'] = bc_outs['strategy'].map(ps.STRATEGY_DICT)
    for g in ps.GENES:
        temp = bc_outs[bc_outs['gene'] == g]
        if type(keep_strats) == str and keep_strats != 'all':
            keep_strats = ['Hyst-BSO: 40']
            
            if g == 'PMS2':
                keep_strats.append('Hyst-BSO: 50')
            else:
                keep_strats.append('Two-Stage Approach')
        else:
            keep_strats = temp['strategy'].to_list()
        temp = temp[temp['strategy'].isin(keep_strats)]
        temp = temp[keep_cols]
        fmt_df = fmt_df.append(temp, ignore_index = True)
    fmt_df['Cancer Incidence'] = fmt_df['Cancer Incidence'].apply(format_percent)
    fmt_df['Cancer Mortality'] = fmt_df['Cancer Mortality'].apply(format_percent)
    fmt_df[ps.QALY_COL] = fmt_df[ps.QALY_COL].apply(format_decimal)
    fmt_df['Cancer Incidence (Mortality)'] = fmt_df['Cancer Incidence'] + ' ('+fmt_df['Cancer Mortality']+')'
    fmt_df.drop(columns = ['Cancer Incidence', 'Cancer Mortality'], inplace=True)
    fmt_df[ps.COST_COL] = fmt_df[ps.COST_COL].apply(format_money)
    fmt_df[ps.ICER_COL] = fmt_df[ps.ICER_COL].apply(format_money)
    fmt_df.to_csv(ps.dump/f"{ps.F_NAME_DICT['BC_FMT_QALYs']}{ps.icer_version}.csv",
                   index = False)
    

        
#create_bc_output_table()    
        
        
        


