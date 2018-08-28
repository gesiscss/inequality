#the following snippet is taken from the github project and the license is creative commons
#https://github.com/oliviaguest/gini
import numpy  as np
import pandas as pd
from scipy.stats import mannwhitneyu
import math

from numpy import std, mean, sqrt


#correct if the population S.D. is expected to be equal for the two groups.

def mann_whitney_effect_size(a, b, alternative='two-sided', effect_formula='r'):
    n1 = len(a)
    n2 = len(b)
    statistic, pvalue = mannwhitneyu(a, b, alternative=alternative)
    z = (statistic-(n1*n2/2))/(math.sqrt(n1*n2*(n1+n2+1)/12))
#     print('n1,n2: ',n1,n2)
#     print("u: ", statistic)
#     print("z: ", z)
#     print("p: ", pvalue)
    if effect_formula == 'r':
        effect = z/math.sqrt(n1+n2)
    elif effect_formula == 'eta':
        effect = (z**2)/(n1+n2-1)
    return abs(effect), statistic, pvalue

def cohen_d(x_means,y_means, x_std, y_std):
    nx = len(x_means)
    ny = len(y_means)
    dof = nx + ny - 2
    res = []
    for i in range(1, nx):      
        res[i] = (x_means[i] - y_means[i]) / sqrt(((nx-1)* x_std[i] ** 2 + (ny-1)* y_std[i] ** 2) / dof)
        i = i+1
    return res

def cliffsD(ser1, ser2):
    # only small integer values!
    np_1 = np.array(ser1, dtype=np.int8)
    np_2 = np.array(ser2, dtype=np.int8)
    return np.mean(np.sign(np_1[:, None] - np_2).mean(axis=1))

def cliffs_delta_cohorts(data, column, val1, val2, criterion, years):
    cliffsD_lst = []
    for start_year in years:
        data1 = data[data.start_year == start_year]
        cliffsd = cliffsD(data1[data1[column] == val1][criterion], data1[data1[column] == val2][criterion])
        cliffsD_lst.append([start_year, cliffsd])
    cliffsD_cohorts = pd.DataFrame(columns=['start_year', 'cliffsD'], data=cliffsD_lst)
    return cliffsD_cohorts

def gini(array):
    
    #"""Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    #array = array.flatten()
    
    try:
        if np.min(array) < 0:
            # Values cannot be negative:
            array -= np.min(array)
    except ValueError:  #raised if `array` is empty.
        return 0
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def groupDataAndCalculateCumulativeValues(data, group_year, criterion):
    
    # Analysis of performance or recognition need not be done only year wise.
    # It can be done in multiples of year - which sometimes makes sense
    # This function helps to group author based on the year passed and accumulates their performance/recognition accordingly
    
    # data - the dataframe containing author publications or citations data
    # group_year - how many years should be clubbed to form
    # criterion - 'num_pub' (or) 'num_cit'
    
    # Group years and associative data and calculates the cumulative value
    if group_year > 1:
        # BUG: Calculation of publications/citations not correct
        # BUG: This code changes the start year parameter! It moves 'maseka lesaoana from 2001 to 2000!
        raise Exception('This part of code is not fixed! Do not use it!')
        
        min_year = data['year'].min() 
        
        # the final year was getting missed out from grouping. Inorder to avoid that the max year is extended by one group year
        # pd.cut() function groups the years and allows the user to assign labels to each group. 
        # It is designed in such a way that it was by default leaving the label for the last group. In order to fix this issue, 
        # an extra bin is added to the last, so that, the second last (which is the actual last with respect to this context) is preserved
        
        # this max_year is used only for grouping - so no worries :-) --> WHY 2*GROUP_YEAR???
        max_year = data['year'].max() + 2*group_year 
        yearGroups = range(min_year, max_year, group_year)
        
        #bins - contains the year groups and labels contains all the group except the last one
        data['year'] = pd.cut(data['year'], bins=yearGroups, labels=yearGroups[:-1],\
                                               include_lowest=True, right=False)
        if 'start_year' in data.columns:
            data['start_year'] = pd.cut(data['start_year'],bins=yearGroups, labels=yearGroups[:-1],\
                                        include_lowest=True, right=False)
            
        if 'end_year' in data.columns:
            data['end_year'] = pd.cut(data['end_year'],bins=yearGroups, labels=yearGroups[:-1],\
                                      include_lowest=True, right=False)

        # within the group - all the multiple efforts need to be joined together
        data[criterion] = data.groupby(['year', 'author'])[criterion].transform(np.sum)

        print("data with duplicates: %s", data.shape[0])
        data = data.drop_duplicates()
        print("data without duplicates: %s", data.shape[0])
        
        print("data with null values: %s", data.shape[0])
        data = data.dropna(how='any')
        print("data without null values: %s", data.shape[0])
        
    #Sort the year and calculate the cumulative values

    data = data.set_index('year').sort_index()
    data['cum_'+criterion] = data.groupby(['author'])[criterion].transform(pd.Series.cumsum)
    data = data.reset_index()
    #print(data.head(10))
    return data




    