# Probability functions

'''
Authors: Elisabeth Silver
Description: Functions to compute probabilities for LS markov model
'''

import math
import numpy as np
import pandas as pd
import presets_lynchGYN as ps
import data_manipulation as dm
from pathlib import Path



# determines if x is between the bounds
def between(x, low_bound, up_bound):
    if x >= low_bound and x < up_bound:
        return True
    else:
        return False

def to_df(arg1, arg2):
    df = pd.DataFrame(list(arg2), index=arg1)
    return df


# Turns rate into a probability
def rate_to_prob(rate, t):
    prob = 1-math.exp(-abs(rate)*t)
#    print(prob)
    return prob

def prob_to_rate(prob, t):
    rate = -(np.log(1-prob))/t
#    print(rate)
    return rate
    

def prob_conv(prob, old_t, new_t):
    temp = prob_to_rate(prob, old_t)
    new_prob = rate_to_prob(temp, new_t)
    return new_prob

# Gives probability based on pw function
def pw_prob(time, slope_array, nodes):
    for n in range(len(nodes)-1):
        if between(time, nodes[n], nodes[n+1]-1):
            prob = rate_to_prob(slope_array[n], 1)
#                                ((nodes[n+1]-nodes[n])*ps.CYCLE_LENGTH))
            print(prob)
            return prob
        elif time >= nodes[len(nodes)-1]:
            prob = rate_to_prob(slope_array[len(slope_array)-1], 1)
#                                ((nodes[len(nodes)-1])*ps.CYCLE_LENGTH))
            return prob

def pw_choose_prob(time, probs, nodes):
    
    for n in range(len(nodes)-1):
        if between(time, nodes[n], nodes[n+1]):
            prob = probs[n]
            #print('between true for', nodes[n], nodes[n+1])
            return prob
        #if age is > upper bound for last prob
        elif time >= nodes[len(nodes)-1]:
            prob = probs[len(probs)-1]
            return prob
        elif time <= nodes[0]:
            #prob = probs[0]
            prob = 0.0
            return prob




def prob_to_prob(prob, from_cycle_length, to_cycle_length, time): 
    # Converts prob per one cycle length to prob per another cycle length
    # Inputs: 
    # prob is list of probabilities at each time point of original cycle length
    # from_cycle_length is original cycle length
    # to_cycle_length is desired cycle length
    # tmin and tmax: minimum and maximum times
    # Output: list of probabilities at each time point at desired cycle length
    
    small_step = 0
    big_step = 0
    t = min(time)
    i = 0
    big_dt = max(from_cycle_length, to_cycle_length)
    small_dt = min(from_cycle_length, to_cycle_length)
    num_cycle_ratio = from_cycle_length / to_cycle_length
    to_prob = []
    
    while t < max(time):
        from_rate = prob_to_rate(prob[i], t)
        to_rate = from_rate / num_cycle_ratio
        to_prob.append(rate_to_prob(to_rate, t))
        if small_step < num_cycle_ratio:
            small_step += 1
        else:
            small_step = 1
            big_step += 1
            i += 1
        t = (big_step*big_dt) + (small_step*small_dt)
#        print(t)
    return to_prob


def discount(cost, t):
    
    # Discounts values over time
    # Inputs: values to discount, length of simulation, and discount rate
    # Output: new array with discounted values
        
    new_costs = cost/((1+ps.d_rate)**t)
        
    return new_costs

def normalize_new(array, row_index):
    i = 0
    other_prob_sum = 0.000
    for i in range(0, len(array)):
        if i != row_index:
            other_prob_sum += array[i]
    if other_prob_sum < 1.00001 and other_prob_sum > 0.99999:
        array[row_index] = 0
    else:
        array[row_index] = 1 - other_prob_sum
    if sum(array) > 1.00001 or sum(array) < 0.99999:
        print('sum != 1')
        
def normalize_checker(array, row_index):
    array_copy = np.copy(array)
    i = 0
    other_prob_sum = 0.
    for i in range(0, len(array_copy)):
        if i != row_index:
            other_prob_sum += array_copy[i]
      
    array_copy[row_index] = 1.000 - other_prob_sum
    if sum(array_copy) > 1.00001:
        print('sum > 1')
        return False
    elif sum(array_copy) < 0.9999:
        print('sum < 1')
        return False
    else:
        return True

# normalizes to 1--for use with death states
def normalize(array, row_index):
    i = 0
    for i in range(0, len(array)):
        if i !=row_index:
            array[i] = 0.0
        else:
            array[i] = 1.0

# noramlizes to a specific number
def normalize_choose(array, number):
    if number == 0:
        print("number is equal to zero")
        return array
    if sum(array) > number:
        array = np.divide(array, sum(array)/number)
#        print(array)
        return array
    elif sum(array) < number:
        array = np.multiply(array, number/sum(array))
#        print(array)
        return array
    else:
        return array

def normalize_switch(array, old_col, new_col):
    i = 0
    other_prob_sum = 0.000
    for i in range(0, len(array)):
        if i == old_col:
            array[old_col] = 0.0
            
        elif i != old_col and i != new_col:
            other_prob_sum += array[i]
        else:
            other_prob_sum += 0.0
      
    array[new_col] = 1 - other_prob_sum
    if sum(array) > 1.00001 or sum(array) < 0.99999:
        print('sum != 1')


def normalize_matrix(matrix):
    matrix_width = matrix.shape[-2]
    matrix_depth = matrix.shape[-3]
    
    for i in range(matrix_depth):
        for j in range(matrix_width):
            matrix[i, j] = normalize(matrix[i, j])
    return matrix
    

def risk_increase(prob, risk):
    inc_prob = prob/(1-risk)
    return inc_prob


def minus(rate_1, rate_2):
# =============================================================================
#     Subtract rate 1 from rate 2
# =============================================================================
    new_rate = rate_2 - rate_1
    return new_rate


def annual_rate_to_prob(rates, age):
    annual_prob = [rate_to_prob(rates[i], 1/minus(age[j], age[j+1]))
                    for i, j in zip(range(len(rates)), 
                                    range(len(age)-1))]
    return annual_prob


def cumul_prob_to_annual(path, gene, col_name):
# =============================================================================
#     Converts cumulative probability to annual probability
# =============================================================================
    if type(col_name) == int:
        col_name = str(int(col_name))
    age, cumul_prob = dm.excel_to_lists(path, gene, col_name)
#    print(cumul_prob)
#    print(age)
    cumul_rate = [prob_to_rate(prob, 1) for prob in cumul_prob]
    annual_rate = [minus(cumul_rate[i], cumul_rate[i+1]) 
                    for i in range(len(cumul_rate)-1)]
    #new_annual_rate = [rate*multiplier for rate in annual_rate]
    annual_prob = [rate_to_prob(annual_rate[i], 1/minus(age[j], age[j+1]))
                    for i, j in zip(range(len(annual_rate)), 
                                    range(len(age)-1))]
    #print(annual_prob)
    #print(age)
    return age, annual_prob, annual_rate

def return_risk(path, gene, col_name):
    age, cumul_prob = dm.excel_to_lists(path, gene, col_name)
    return cumul_prob[len(cumul_prob)-1]
#FIXME
def cumul_prob_to_annual_from_df(risk_df, col_name):
# =============================================================================
#     Converts cumulative probability to annual probability
# =============================================================================
    age = risk_df['age'].values()
    cumul_prob = risk_df[col_name].values()
#    print(cumul_prob)
#    print(age)
    cumul_rate = [prob_to_rate(prob, 1) for prob in cumul_prob]
    annual_rate = [minus(cumul_rate[i], cumul_rate[i+1]) 
                    for i in range(len(cumul_rate)-1)]
    #new_annual_rate = [rate*multiplier for rate in annual_rate]
    annual_prob = [rate_to_prob(annual_rate[i], 1/minus(age[j], age[j+1]))
                    for i, j in zip(range(len(annual_rate)), 
                                    range(len(age)-1))]
    #print(annual_prob)
    #print(age)
    return age, annual_prob, annual_rate


def check_valid_file(filename):
    config = Path(filename)
    if config.is_file():
        #print('file is valid')
        return True
    else:
        #print('file is not valid')
        return False

    


