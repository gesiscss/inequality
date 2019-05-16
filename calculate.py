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
    # http://yatani.jp/teaching/doku.php?id=hcistats:mannwhitney#effect_size
    # http://core.ecu.edu/psyc/wuenschk/docs30/Nonparametric-EffectSize.pdf
    # https://myadm2014.files.wordpress.com/2017/02/spss-survival-manual-a-step-by-step-guide-to-data-analysis-using-spss-for-windows-3rd-edition-aug-2007-2.pdf
    # https://www.psychometrica.de/effect_size.html
    if effect_formula == 'r':
        effect = z/math.sqrt(n1+n2)
    elif effect_formula == 'eta':
        effect = (z**2)/(n1+n2-1)
    elif effect_formula == 'common_language':
        effect = statistic/(n1*n2)
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

def coef_var(array):
    return np.std(array)/np.mean(array)


def percentage_zeros(array):
    return ((array.size)-np.count_nonzero(array))/(array.size)

def hhi(array):
    # Herfindahl Hirschman Index   
    # formula not good
    # TODO check formula
    total = np.sum(array)
    if total == 0: return 0
    N = np.count_nonzero(array)
    if N == 1: return 1
    H = np.sum([(value/total)**2 for value in array])
    return (H- 1/N)/(1- 1/N)

def gini(array):
    array_copy = np.copy(array)
    #"""Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    #array = array.flatten()
    
    try:
        if np.min(array_copy) < 0:
            # Values cannot be negative:
            array_copy -= np.min(array_copy)
    except ValueError:  #raised if `array` is empty.
        return 0
    # Values cannot be 0:
    array_copy += 0.0000001
    # Values must be sorted:
    array_copy = np.sort(array_copy)
    # Index per array element:
    index = np.arange(1,array_copy.shape[0]+1)
    # Number of array elements:
    n = array_copy.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array_copy)) / (n * np.sum(array_copy)))


def gini_nonzero(array):
    array_copy = np.copy(array)
    #"""Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    #array = array.flatten()
    
    try:
        if np.min(array_copy) < 0:
            # Values cannot be negative:
            array_copy -= np.min(array_copy)
    except ValueError:  #raised if `array` is empty.
        return 0
    # Remove zeros: #Daniel replaced trim_zeros with nonzero
    array_copy = array_copy[array_copy.nonzero()]
    # Values must be sorted:
    array_copy = np.sort(array_copy)
    # Index per array element:
    index = np.arange(1,array_copy.shape[0]+1)
    # Number of array elements:
    n = array_copy.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array_copy)) / (n * np.sum(array_copy)))


def calculate_cumulative_for_authors(data, criterion):
    # data - the dataframe containing author publications or citations data
    # criterion - 'num_pub' (or) 'num_cit'
    # Group years and associative data and calculates the cumulative value

    data = data.set_index('year').sort_index()
    data['cum_'+criterion] = data.groupby(['author'])[criterion].transform(pd.Series.cumsum)
    data = data.reset_index()

    return data




    