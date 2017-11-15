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
    GROUP_YEAR = group_year

    if GROUP_YEAR > 1:
        MIN_YEAR = data['year'].min() 
        MAX_YEAR = data['year'].max() + 2*GROUP_YEAR
        yearGroups = range(MIN_YEAR, MAX_YEAR, GROUP_YEAR)
        data['year'] = pd.cut(data['year'], bins=yearGroups, labels=yearGroups[:-1],\
                                               include_lowest=True, right=False)
        data['start_year'] = pd.cut(data['start_year'],bins=yearGroups, labels=yearGroups[:-1],\
                                               include_lowest=True, right=False)
        data['end_year'] = pd.cut(data['end_year'],bins=yearGroups, labels=yearGroups[:-1],\
                                               include_lowest=True, right=False)

        data[criterion] = data.groupby(['year', 'author'])[criterion].transform(np.sum)

        print("data with duplicates: %s", data.shape[0])
        data = data.drop_duplicates()
        print("data without duplicates: %s", data.shape[0])

    #print(data[data['author'] == 'donatella sciuto'].head(10))

    data = data.set_index('year').sort_index()
    data['cum_'+criterion] = data.groupby(['author'])[criterion].transform(pd.Series.cumsum)
    data = data.reset_index()
    print(data.head(10))
    return data
    