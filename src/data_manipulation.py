# functions that are useful for data manipulation
import numpy as np
import pandas as pd
import pathlib as pl

def csv_to_lists(path):
    df = pd.read_csv(path)
    x_data = df.iloc[:,0]
    y_data = df.iloc[:,1]

    # convert from data frame to list
    x = x_data.values.tolist()
    y = y_data.values.tolist()
    
    return x, y



def cell_to_list(cell_str):
    if type(cell_str) != str:
        return cell_str
    new_str = cell_str.split(',')
    new_list = [float(num) for num in new_str]
    
    return new_list



def excel_to_lists(path, sheet_name, col_name):
    df = pd.read_excel(path, sheet_name)
    x_data = df.iloc[:,0]
    try:
        y_data = df.loc[:, col_name]
    except:
        y_data = df.loc[:, str(int(col_name))]

    # convert from data frame to list
    x = x_data.values.tolist()
    y = y_data.values.tolist()

    return x, y

def flip(dictionary):
    new_dict = dict((v, k) for k, v in dictionary.items())
    return new_dict

 
def selection(df, keywords, column):
    df_new = df.loc[df[column].isin(keywords)]
    return df_new


def exclusion(df, keywords, column):
    df_new = df.loc[~df[column].isin(keywords)]
    return df_new


def keyword_search(df, array, column):
    keywords = []
    for i in array:
        for j in df[column]:
            print(j)
            if i.casefold() in str(j).casefold():
                keywords.append(j)
    df_new = selection(df, keywords, column)
    return df_new

        
        
