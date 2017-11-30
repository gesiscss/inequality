#the following snippet is taken from the github project and the license is creative commons
#https://github.com/oliviaguest/gini
import numpy  as np
import pandas as pd
def gini(array):
    #"""Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.min(array) < 0:
        # Values cannot be negative:
        array -= np.min(array)
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
    # data - the dataframe containing author publications or citations data
    # group_year - how many years should be clubbed to form
    # criterion - 'num_pub' (or) 'num_cit'
    
    # Group years and associative data and calculates the cumulative value
    if group_year > 1:
        min_year = data['year'].min() 
        # the final year was getting missed out from grouping. Inorder to avoid that the max year is extended by one group year
        # pd.cut() function groups the years and allows the user to assign labels to each group. 
        # It is designed in such a way that it was by default leaving the label for the last group. In order to fix this issue, 
        # an extra bin is added to the last, so that, the second last (which is the actual last with respect to this context) is preserved
        
        # this max_year is used only for grouping - so no worries :-)
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
    