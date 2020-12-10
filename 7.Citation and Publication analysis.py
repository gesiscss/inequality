# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# user defined modules
import calculate#, plot
from calculate import gini, hhi, percentage_zeros, gini_nonzero
from credible_authors_ import DataStore

import warnings
warnings.filterwarnings("ignore")

# Configuration - needed?
gini_nonzero.display_name = 'Gini without zeroes'
percentage_zeros.display_name = 'Percentage of zeroes'
np.mean.display_name = 'Average'

START_YEAR = 1970
LAST_START_YEAR = 2000

# +
# from importlib import reload
# reload(calculate)
# reload(plot)
# -

color_full = ['#000000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
color_pale = ['#7f7f7f', '#f18c8d', '#9bbedb', '#a6d7a4', '#cba6d1', '#ffbf7f', '#ffff99', '#d2aa93', '#fbc0df', '#cccccc']

# ## Load data from csv

data_store = DataStore()
all_papers_num_cit = data_store.all_papers_num_cit

data_store.credible_authors.shape

data_store.authorPublicationData.pub_id.nunique()

data_store.authorCitationsData.shape

data_store.authorCitationsData.year.min()

data_store.authorPublicationData.year.min()

author_order = pd.read_csv('derived-data/publication_authors_order_2017.csv')

credible_authors = pd.read_csv('derived-data/authors-scientific-extended_all.csv')

len(credible_authors.columns)

# ### Window counts

# Load in citation window data:

WINDOW_SIZE = 3
citations_window = pd.read_csv(f'derived-data/citations_window_{WINDOW_SIZE}.csv')
citations_window_first = pd.read_csv(f'derived-data/citations_window_{WINDOW_SIZE}_first.csv')

# +
# WINDOW_SIZE = 5
# citations_window_5 = pd.read_csv(bigdata + f'derived-data/citations_window_{WINDOW_SIZE}.csv')
# citations_window_first_5 = pd.read_csv(bigdata + f'derived-data/citations_window_{WINDOW_SIZE}_first.csv')

# +
dropout_cols = ['author', 'dropped_after_10', 'dropped_after_5']
citations_window = citations_window.merge(credible_authors[dropout_cols], on='author', how='left')
citations_window_first = citations_window_first.merge(credible_authors[dropout_cols], on='author', how='left')

# citations_window_5 = citations_window_5.merge(credible_authors[dropout_cols], on='author', how='left')
# citations_window_first_5 = citations_window_first_5.merge(credible_authors[dropout_cols], on='author', how='left')
# -

# ### Citations and uncited papers
# Dataset with all the citations to papers, together with papers that received no citations (cit_id is null)

uncited_papers_network = data_store.authorPublicationData.merge(data_store.authorCitationsData, left_on='pub_id',
                                                                     right_on='id2', how='left',
                                                                     suffixes=('_pub', '_cit'))

# by merging here on 'inner', we remove duplicates with many authors
uncited_papers_network_first_auth = uncited_papers_network.merge(author_order[['first_author', 'pub_id']], 
                                                                 left_on=['author', 'pub_id'], 
                                                                 right_on=['first_author', 'pub_id'],how='inner')

# add start year
uncited_papers_network_first_auth = uncited_papers_network_first_auth.merge(data_store.credible_authors[
    ['author', 'start_year']], on='author', how='left')

uncited_papers_network_first_auth.drop('id2', axis='columns', inplace=True)
uncited_papers_network_first_auth.rename({'id1':'cit_id'}, axis='columns', inplace=True)

career_len = data_store.credible_authors[['author', 'career_length']]
# career_len_10 = career_len[career_len['career_length'] >= 10]


# ### UnCited papers network - first author

# uncited_papers_network_first_auth
# author == first_author
# contains all papers, with their first authors. 
# duplicate rows for pub_id are different citations, with cit_id for the citing paper
# cit_id == NaN -> paper never cited
uncited_papers_network_first_auth.head()

# aggregate citations, and get the count - per author, per year pub, per paper, per year cited
auth_year_pub_cit_count = uncited_papers_network_first_auth.groupby(
    ['author', 'year_pub', 'pub_id', 'year_cit']).agg({'cit_id': 'count'})
auth_year_pub_cit_count = auth_year_pub_cit_count.reset_index()

# add start year
auth_year_pub_cit_count = auth_year_pub_cit_count.merge(data_store.credible_authors[['author', 'start_year']], 
                                                        on='author', how='left')

# remove cited before published. Could be right, but makes the graphs weird
auth_year_pub_cit_count = auth_year_pub_cit_count[auth_year_pub_cit_count.year_cit >= auth_year_pub_cit_count.year_pub]

# take the best paper in a specific year cited
agg_cit_per_auth_year = auth_year_pub_cit_count.groupby(['start_year', 'author', 'year_pub', 'year_cit']).agg(
    {'cit_id': 'max'}).reset_index()

# take the average in a specific year cited
agg_cit_per_auth_year_avg = auth_year_pub_cit_count.groupby(['start_year', 'author', 'year_pub', 'year_cit']).agg(
    {'cit_id': 'mean'}).reset_index()

# veryfied this df for correctness with a different calc method
cohort_size_first_auth = uncited_papers_network_first_auth.groupby('start_year')['author'].nunique()
cohort_size_first_auth = cohort_size_first_auth.reset_index()
cohort_size_first_auth.rename({'author':'cohort_size'}, axis=1, inplace=True)

# add start year
agg_cit_per_auth_year = agg_cit_per_auth_year.merge(cohort_size_first_auth, on='start_year', how='left')
agg_cit_per_auth_year_avg = agg_cit_per_auth_year_avg.merge(cohort_size_first_auth, on='start_year', how='left')

agg_cit_per_auth_year = agg_cit_per_auth_year[(agg_cit_per_auth_year.start_year >= 1970) &
                                              (agg_cit_per_auth_year.start_year <=2000)]
agg_cit_per_auth_year_avg = agg_cit_per_auth_year_avg[(agg_cit_per_auth_year_avg.start_year >= 1970) &
                                              (agg_cit_per_auth_year_avg.start_year <=2000)]

# #### Data interrogation 

uncited_papers_network_first_auth.dropna().cit_id.count()

uncited_papers_network_first_auth.year_pub.min()

# author_order[author_order.pub_id == 'a43af2b6-93e5-480f-9678-8394483315a8']
uncited_papers_network_first_auth[uncited_papers_network_first_auth.pub_id == 'a43af2b6-93e5-480f-9678-8394483315a8']

uncited_papers_network_first_auth['pub_id'].nunique()

auth_year_pub_cit_count[(auth_year_pub_cit_count.start_year == 1999) & 
                        (auth_year_pub_cit_count.year_pub == 1999) &
#                         (auth_year_pub_cit_count.year_cit == 1999) &
                        (auth_year_pub_cit_count.author == 'joseph mitola iii')].sort_values('year_cit')

# these numbers are right, according to the combined df in 0.Data Preproc.
# this means i am grabbing all the citations by all the papers, and that the filters are fixed
agg_cit_per_auth_year[(agg_cit_per_auth_year.start_year == 1999) & 
                        (agg_cit_per_auth_year.year_pub == 1999) &
#                         (agg_cit_per_auth_year.year_cit == 1999) &
                        (agg_cit_per_auth_year.author == 'joseph mitola iii')].sort_values('year_cit')

# number of first authors in 1994
uncited_papers_network_first_auth[uncited_papers_network_first_auth.start_year == 1994]['author'].nunique()

agg_cit_per_auth_year[(agg_cit_per_auth_year.start_year == 1994)]['author'].nunique()# & (agg_cit_per_auth_year.year_cit == 1994)]

# i guess also all authors in 1994
auths_1994 = agg_cit_per_auth_year[(agg_cit_per_auth_year.start_year == 1994) & 
                      (agg_cit_per_auth_year.year_cit == 1994)]['author'].unique()
print(auths_1994.shape)

author_order = author_order.merge(data_store.credible_authors[
    ['author', 'start_year']], left_on='first_author', right_on='author', how='left')

# all first authors in 1994
all_auths_1994 = author_order[author_order.start_year==1994]['author'].unique()
print(all_auths_1994.shape)

# ### top k percent 

agg_cit_per_auth_year['cit_id'] = agg_cit_per_auth_year['cit_id'].astype(int)


# TODO: Flaw: array is not full. Taking k percent of non zeros. Add the missing zeros
def perc_owned_by_top_k(arr, size, k):
    arr = arr.values
    top_k = int(round((size/100)*k))
    return sum(arr[arr.argsort()[-top_k:]])/sum(arr)


k=1
perc_owned = agg_cit_per_auth_year.groupby(['start_year', 'year_pub', 'year_cit'])['cit_id', 'cohort_size'].apply(
    lambda x: perc_owned_by_top_k(x['cit_id'], x['cohort_size'].max(), k)).reset_index()
perc_owned.rename({0: 'perc_owned'}, axis=1, inplace=True)

data = perc_owned
metric = 'perc_owned'
for start_year in data.start_year.unique()[-10::2]:
    cohort_data = data[data.start_year == start_year]
    for year_pub in cohort_data.year_pub.unique()[:11:3]:
        one_year = cohort_data[cohort_data.year_pub == year_pub]
        plt.plot(range(0,5), one_year[metric][:5], label=f'Pubs in {year_pub}')
    plt.ylabel(f'Percent of citations held by top {k}perc')
    plt.xlabel('T - time after publishing')
    plt.title(f'Cohort start year: {start_year}')
    plt.legend()
    plt.show()


# ## Gini across cohorts

def aggregate_cohort_data(citations_window, func):
    gini_cohorts_ca = citations_window.groupby(['start_year', 'career_age']).agg({
        'num_pub': func,
        'num_cit': func,
        'cum_num_pub': func,
        'cum_num_cit': func}).reset_index()
    return gini_cohorts_ca


gini_cohorts_ca = aggregate_cohort_data(citations_window, gini)
gini_cohorts_ca_first = aggregate_cohort_data(citations_window_first, gini)


def plot_criteria_over_cohorts(data, criteria, criteria_name, title, letter, legend=True, name_ext=''):
    linewidth = 2
    fontsize = 18
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    color = ['#000000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    career_ages = [15,10,5,3]
    for i in range(0, len(career_ages)):
        df = data[data['career_age'] == career_ages[i]]
        ax.plot(df['start_year'], df[criteria], linewidth=linewidth, label=career_ages[i], color=color[i])
#     ax.set_xlim([0.25, 15.75])
    if('cum' in criteria): ax.set_ylim([0.1, 1.05])
    else: ax.set_ylim([0.15, 1.05])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('both')
#     ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel('Cohort start year', fontsize=fontsize)
    ax.set_ylabel(f'{criteria_name}', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks([1970, 1980, 1990, 2000])
    ax.tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize, pad=7)
    ax.tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize, pad=7)
    ax.tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
    ax.tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    if legend: ax.legend(fontsize=fontsize-6)
    plt.gcf().text(0., 0.9, letter, fontsize=fontsize*2)
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.2, top=0.9)
    fig.savefig(f'./fig-7-notebook/{criteria}_over_cohorts{name_ext}.pdf')


# +
def plot_for_latex3(citations_window, citations_window_first):
    # all authors
    gini_cohorts_ca = aggregate_cohort_data(citations_window, gini)
    plot_array_configs2(gini_cohorts_ca, get_config2('gini', 'Gini'), letters11)
    # first author
    gini_cohorts_ca_first = aggregate_cohort_data(citations_window_first, gini)
    plot_array_configs2(gini_cohorts_ca_first, get_config2('gini', 'Gini'), letters12, name_ext='_first')
    # dropouts removed 10y
    citations_window_stayed = citations_window[citations_window.dropped_after_10 == False]
    gini_cohorts_ca_stayed = aggregate_cohort_data(citations_window_stayed, gini)
    plot_array_configs2(gini_cohorts_ca_stayed, get_config2('gini', 'Gini'), letters21, name_ext='_stay10')
    # dropouts removed first auth 10y
    citations_window_stayed_first = citations_window_first[citations_window_first.dropped_after_10 == False]
    gini_cohorts_ca_stayed_first = aggregate_cohort_data(citations_window_stayed_first, gini)
    plot_array_configs2(gini_cohorts_ca_stayed_first, get_config2('gini', 'Gini'), letters22, name_ext='_stay10_first')
    # dropouts removed 5y
#     citations_window_stayed_5 = citations_window[citations_window.dropped_after_5 == False]
#     gini_cohorts_ca_stayed_5 = aggregate_cohort_data(citations_window_stayed_5, gini)
#     plot_array_configs2(gini_cohorts_ca_stayed_5, get_config2('gini', 'Gini'), letters31, name_ext='_stay5')
    # dropouts removed first auth 5y
#     citations_window_stayed_first_5 = citations_window_first[citations_window_first.dropped_after_5 == False]
#     gini_cohorts_ca_stayed_first_5 = aggregate_cohort_data(citations_window_stayed_first_5, gini)
#     plot_array_configs2(gini_cohorts_ca_stayed_first_5, get_config2('gini', 'Gini'), letters32, name_ext='_stay5_first')
    
def plot_for_latex4(citations_window, citations_window_first):
    # active in first 3 years
    act1 = citations_window[((citations_window['career_age']==1) & (citations_window['num_pub']>0))].author
    act2 = citations_window[((citations_window['career_age']==2) & (citations_window['num_pub']>0))].author
    act3 = citations_window[((citations_window['career_age']==3) & (citations_window['num_pub']>0))].author

    act123 = list(set(act1).intersection(set(act2)).intersection(act3))
    citations_window_active = citations_window[citations_window.author.isin(act123)]
    gini_cohorts_ca_active = aggregate_cohort_data(citations_window_active, gini)
    plot_array_configs2(gini_cohorts_ca_active, get_config2('gini', 'Gini'), letters31, name_ext='_active')
    
    citations_window_active_first = citations_window_first[citations_window_first.author.isin(act123)]
    gini_cohorts_ca_active_first = aggregate_cohort_data(citations_window_active_first, gini)
    plot_array_configs2(gini_cohorts_ca_active_first, get_config2('gini', 'Gini'), letters32, name_ext='_active_first')
    
plot_for_latex3(citations_window, citations_window_first)
plot_for_latex4(citations_window, citations_window_first)
# -

# Export derived data:

gini_cohorts_ca.head()
gini_cohorts_ca.to_csv(bigdata + 'derived-data/gini_cohorts_ca.csv', encoding='utf-8', sep='\t', index=False)

gini_cohorts_ca_first.head()
gini_cohorts_ca_first.to_csv(bigdata + 'derived-data/gini_cohorts_ca_first.csv', encoding='utf-8', sep='\t', index=False)

citations_window_stayed = citations_window[citations_window.dropped_after_10 == False]
gini_cohorts_ca_stayed = aggregate_cohort_data(citations_window_stayed, gini)
gini_cohorts_ca_stayed.head()
gini_cohorts_ca_stayed.to_csv(bigdata + 'derived-data/gini_cohorts_ca_stayed.csv', encoding='utf-8', sep='\t', index=False)

citations_window_stayed_first = citations_window_first[citations_window_first.dropped_after_10 == False]
gini_cohorts_ca_stayed_first = aggregate_cohort_data(citations_window_stayed_first, gini)
gini_cohorts_ca_stayed_first.head()
gini_cohorts_ca_stayed_first.to_csv(bigdata + 'derived-data/gini_cohorts_ca_stayed_first.csv', encoding='utf-8', sep='\t', index=False)

# +
act1 = citations_window[((citations_window['career_age']==1) & (citations_window['num_pub']>0))].author
act2 = citations_window[((citations_window['career_age']==2) & (citations_window['num_pub']>0))].author
act3 = citations_window[((citations_window['career_age']==3) & (citations_window['num_pub']>0))].author

act123 = list(set(act1).intersection(set(act2)).intersection(act3))
citations_window_active = citations_window[citations_window.author.isin(act123)]
gini_cohorts_ca_active = aggregate_cohort_data(citations_window_active, gini)
gini_cohorts_ca_active.head()
gini_cohorts_ca_active.to_csv(bigdata + 'derived-data/gini_cohorts_ca_active.csv', encoding='utf-8', sep='\t', index=False)
# -

gini_cohorts_ca_active_first = citations_window_first[citations_window_first.author.isin(act123)]
gini_cohorts_ca_active_first = aggregate_cohort_data(citations_window_active_first, gini)
gini_cohorts_ca_active_first.head()
gini_cohorts_ca_active_first.to_csv(bigdata + 'derived-data/gini_cohorts_ca_active_first.csv', encoding='utf-8', sep='\t', index=False)

# ### Dropout in the first 5 years

df_dropped_cohorts = citations_window[citations_window.career_age==1].groupby(['start_year']).agg(
    {'dropped_after_10':['sum', 'count']
    })
df_dropped_cohorts.columns = ['Dropped', 'All authors']

df_dropped_cohorts['percent'] = df_dropped_cohorts['Dropped'] / df_dropped_cohorts['All authors'] * 100

# +
fig, ax2 = plt.subplots()
ax1 = ax2.twinx()

ax1.fill_between(df_dropped_cohorts.index, df_dropped_cohorts['All authors'], color="#cedebd", alpha=1, label='All authors')
ax1.plot(df_dropped_cohorts.index, df_dropped_cohorts['All authors'], color="#411f1f", alpha=0.6)

ax1.fill_between(df_dropped_cohorts.index, df_dropped_cohorts['Dropped'], color="#86c4ba", alpha=1, label='Dropped out')
ax1.plot(df_dropped_cohorts.index, df_dropped_cohorts['Dropped'], color="#411f1f", alpha=0.6)

ax2.plot(df_dropped_cohorts['percent'], color='#1f4068', label='Percentage')
ax2.set_ylim([40, 80])
plt.title('Dropouts')
ax1.set(xlabel='Cohort start year', ylabel='Author count')
ax2.set(ylabel='Dropout percent')

ax2.set_zorder(2)
ax1.set_zorder(1)
ax2.patch.set_visible(False)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=9)
fig.savefig(f'./fig-7-notebook/dropouts_cohorts.png')
fig.show()
# -

df_dropped_malefemale = citations_window[(citations_window.gender.isin(['m', 'f'])) & (citations_window.career_age==1)]

df_dropped_combined = df_dropped_malefemale.groupby(['start_year']).agg({'dropped_after_10':['sum', 'count']})
df_dropped_combined.columns = ['Dropped', 'All authors']
df_dropped_mf = df_dropped_malefemale.groupby(['start_year', 'gender']).agg({'dropped_after_10':['sum', 'count']})
df_dropped_mf.columns = ['Dropped', 'All authors']
df_dropped_mf = df_dropped_mf.reset_index()
df_dropped_mf = df_dropped_mf.set_index('start_year')

# +
df_dropped_m = df_dropped_mf[df_dropped_mf.gender=='m']
df_dropped_f = df_dropped_mf[df_dropped_mf.gender=='f']

df_dropped_m['percent'] = df_dropped_m['Dropped'] / df_dropped_m['All authors'] * 100
df_dropped_f['percent'] = df_dropped_f['Dropped'] / df_dropped_f['All authors'] * 100

# +
fig, ax2 = plt.subplots()
ax1 = ax2.twinx()
# all
ax1.fill_between(df_dropped_combined.index, df_dropped_combined['All authors'], color="#e4e3e3", alpha=1, label='All authors')
ax1.plot(df_dropped_combined.index, df_dropped_combined['All authors'], color="#411f1f", alpha=0.6)
# male
ax1.fill_between(df_dropped_m.index, df_dropped_m['Dropped'], color="#3282b8", alpha=1, label='Male dropped')
ax1.plot(df_dropped_m.index, df_dropped_m['Dropped'], color="#411f1f", alpha=0.6)
# female
ax1.fill_between(df_dropped_f.index, df_dropped_f['Dropped'], color="#ff847c", alpha=1, label='Female dropped')
ax1.plot(df_dropped_f.index, df_dropped_f['Dropped'], color="#411f1f", alpha=0.6)

ax2.plot(df_dropped_m['percent'], color='#1f4068', label='Percentage male')
ax2.set_ylim([35, 95])
ax2.plot(df_dropped_f['percent'], color='#d54062', label='Percentage female')

plt.title('Dropouts')
ax1.set(xlabel='Cohort start year', ylabel='Author count')
ax2.set(ylabel='Dropout percent')

ax2.set_zorder(2)
ax1.set_zorder(1)
ax2.patch.set_visible(False)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=9)
fig.savefig(f'./fig-7-notebook/dropouts_gender.png')
fig.show()

# +
# x axis: max absence years
# y axis: women with x amount of abs years divided by num women
# same style as the prev plot for dropout
# -

credible_authors_mf = credible_authors[credible_authors.gender.isin(['m', 'f'])]
credible_authors_mf = credible_authors_mf[['gender','max_absence_0_15']]

male_absence = credible_authors_mf[credible_authors_mf.gender == 'm']['max_absence_0_15']
female_absence = credible_authors_mf[credible_authors_mf.gender == 'f']['max_absence_0_15']

female_absence

percent_f

female_absence_pos

percent_f.values()

# +
from collections import Counter
male_absence = Counter(male_absence)
female_absence = Counter(female_absence)

total_m = sum(male_absence.values())
total_f = sum(female_absence.values())
percent_m = {key: value*100/total_m for key, value in male_absence.items()}
percent_f = {key: value*100/total_f for key, value in female_absence.items()}
# -

percent_df_m = pd.DataFrame.from_dict(percent_m, orient='index', columns=['val']).reset_index()
percent_df_f = pd.DataFrame.from_dict(percent_f, orient='index', columns=['val']).reset_index()

# +
width = 0.45  # the width of the bars
percent_df_m['index'] = percent_df_m['index'] - width/2
percent_df_f['index'] = percent_df_f['index'] + width/2

fig, ax = plt.subplots()
rects1 = ax.bar(percent_df_m['index'], percent_df_m['val'], width, label='Men', color='C0')
rects2 = ax.bar(percent_df_f['index'], percent_df_f['val'], width, label='Women', color='C3')

# plt.plot(male_absence_pos, list(percent_m.values()))
# plt.plot(female_absence_pos, list(percent_f.values()))

ax.set_ylabel('Number of authors')
ax.set_title('Maximum absence by gender')
ax.set_xticks(male_absence_pos)
ax.legend()


# -

# ### gini

def gini_expand(arr, size):
#     print(len(arr), size)
    num_zer = size - len(arr)
    arr_full = np.append(arr, np.zeros(num_zer))
    return gini(arr_full)


def get_expanded_gini_df(data):
    df_gini = data.groupby(['start_year', 'year_pub', 'year_cit'])['cit_id', 'cohort_size'].apply(
        lambda x: gini_expand(x['cit_id'], x['cohort_size'].max())).reset_index()
    df_gini.rename({0: 'gini'}, axis=1, inplace=True)
    return df_gini


# +
df_gini_max = get_expanded_gini_df(agg_cit_per_auth_year)
df_gini_avg = get_expanded_gini_df(agg_cit_per_auth_year_avg)

df_gini_max['career_age'] = df_gini_max['year_pub'] - df_gini_max['start_year'] + 1
df_gini_avg['career_age'] = df_gini_avg['year_pub'] - df_gini_avg['start_year'] + 1


# -

# The inequality is very big because of the granularity of this chart. 
# We first divide people into cohorts. Then for different career agees, we divide the paper's citations across Time after publishing, T. This makes the data very sparse, and we end up with a huge number of zeroes. <br>
# Example: <br>
# Published in 1991, cited in 1992 and 1995 <br>
# Array: [0, X, 0, 0, Y]

# + {"code_folding": []}
def plot_ineq_(data, metric, ext):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    i = 0
    flat_axes = list(axs.flat)
    for start_year in [1970, 1980, 1990, 2000]:
        ax = flat_axes[i]
        cohort_data = data[data.start_year == start_year]
        for year_pub in cohort_data.year_pub.unique()[:11:3]:
            one_year = cohort_data[cohort_data.year_pub == year_pub]
            ax.plot(range(0,5), one_year[metric][:5], label=f'Career age {year_pub-start_year+1}')
        ax.set_title(f'Cohort start year: {start_year}')
        ax.legend()
        i+=1
    for ax in axs.flat:
        ax.set(xlabel='T - time after publishing', ylabel='Gini over citations')
    fig.suptitle(f"Aggregation metric: {ext}")
    fig.savefig(f'./fig-7-notebook/first_auth_ineq_gini_{ext}.pdf')
    fig.show()
# plot_ineq_(df_gini_max, 'gini', 'max')
# plot_ineq_(df_gini_avg, 'gini', 'avg')


# -

# 5 year df for max
df_gini_max_y5 = df_gini_max[df_gini_max.year_cit == df_gini_max.year_pub + 4]
career_ages = [1,2,4,7,10]
df_gini_max_y5 = df_gini_max_y5[df_gini_max_y5['career_age'].isin(career_ages)]

# 5 year df for max
df_gini_avg_y5 = df_gini_avg[df_gini_avg.year_cit == df_gini_avg.year_pub + 4]
career_ages = [1,2,4,7,10]
df_gini_avg_y5 = df_gini_avg_y5[df_gini_avg_y5['career_age'].isin(career_ages)]


def plot_first_auth_ineq_gini(data_df, param):
    career_ages = [1,2,4,7,10]
    career_ages = [str(x) for x in career_ages]
    start_years_gini = [1970, 1980, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]
    data = data_df[data_df.start_year.isin(start_years_gini)]

    fig, axs = plt.subplots(2, 6, sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(15,7))

    ax_outside = fig.add_subplot(111, frameon=False)
    ax_outside.grid(False)
    ax_outside.axes.get_xaxis().set_ticks([])
    ax_outside.axes.get_yaxis().set_ticks([])

    markers = ['o', 'x', '.', '+', '^']

    for ax, year in zip(axs.flat, start_years_gini):
        x = list(range(5))
        y = data[data.start_year == year]['gini']
        for marker, x_pos, y_val in zip(markers, x, y):
            ax.scatter(x_pos, y_val, marker=marker, label=f'Career age {career_ages[x_pos]}')
        ax.set_xlabel(year)
        ax.set_xticks(x)

    for ax in axs.flat:
    #     ax.label_outer()
        ax.grid(False)
        ax.set_xticklabels(career_ages)
    for ax in axs.flat[:6]:
        ax.xaxis.set_label_position('top') 
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.84,0.22))

    ax_outside.set_xlabel('Cohort start year', labelpad=50, fontweight='bold')
    ax_outside.set_ylabel(f'Gini inequality five years after publishing - {param.upper()}', labelpad=50, fontweight='bold')
    fig.savefig(f'./fig-7-notebook/first_auth_ineq_gini_{param}_squeze.pdf')
plot_first_auth_ineq_gini(df_gini_max_y5, 'avg')
plot_first_auth_ineq_gini(df_gini_max_y5, 'max')

# ### Cliffs Delta

# +
cum_cit_pub_agg_first = citations_window_first.groupby(['start_year', 'career_age', 'gender']).agg({
    'cum_num_pub': list,
    'cum_num_cit': list
}).reset_index()

cum_cit_pub_agg = citations_window.groupby(['start_year', 'career_age', 'gender']).agg({
    'cum_num_pub': list,
    'cum_num_cit': list
}).reset_index()


# + {"code_folding": []}
def get_cohort_effect_size(cohort_careerage_df, metric, gen_a='m', gen_b='f', eff_form='r'):
    data = cohort_careerage_df[cohort_careerage_df.gender.isin([gen_a, gen_b])]
    data = data.set_index(['start_year', 'career_age', 'gender']).unstack(level=-1)
    data.columns = ['_'.join(col) for col in data.columns]
    data['cliffd_m_f'] = data.apply(lambda x: calculate.cliffsD(x[f'{metric}_{gen_a}'], x[f'{metric}_{gen_b}']), axis=1)
    mwu = data.apply(lambda x: calculate.mann_whitney_effect_size(
        x[f'{metric}_{gen_a}'], x[f'{metric}_{gen_b}'], effect_formula=eff_form), axis=1).apply(pd.Series)
    mwu.columns = ['effect', 'statistic', 'pvalue']
    data = data.join(mwu)
    data = data[['cliffd_m_f', 'effect', 'statistic', 'pvalue']]
    data = data.reset_index()
    return data    


# + {"code_folding": []}
def plot_cohort_diffs_over_ages(stats, criterion, criterion_display, ext='', remove_half=False):
    # x - career age
    i = 0  # to point to the right figure
    j = 0
    linewidth = 2
    fontsize = 18

    # rearange subplots dynamically
    cols = 3
    fig_size = 7
    cohort_start_years = np.unique(stats["start_year"].values)
    # remove 1970 from plot
    cohort_start_years = cohort_start_years[1:]
    if remove_half:
        # remove every second year
        cohort_start_years = cohort_start_years[1::2]
    # 15 cohorts?
    num_coh = len(cohort_start_years)
    if (num_coh > 20):
        cols = 6
        fig_size = 14

    nrows = math.ceil(float(len(cohort_start_years)) / float(cols))
    nrows = int(nrows)

    fig3, ax3 = plt.subplots(nrows, cols, sharex=True, sharey=True, figsize=(fig_size, 10))
    fig3.tight_layout()
    # Create a big subplot to created axis labels that scale with figure size
    ax_outside = fig3.add_subplot(111, frameon=False)
    ax_outside.grid(False)
    # hide tick and tick label of the big axes
    ax_outside.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax_outside.set_xlabel('Career Age', labelpad=20, fontsize=fontsize)
    ax_outside.set_ylabel(f'Cliffs Delta', labelpad=20, fontsize=fontsize)

    for year in cohort_start_years:
        cohort = stats[stats["start_year"] == year]
        
        sig_male = cohort[(cohort['cliffd_m_f']>=0.1)].shape[0]# & (cohort.pvalue <= 0.05)
        sig_female = cohort[(cohort['cliffd_m_f']<=-0.1)].shape[0]# & (cohort.pvalue <= 0.05)
        significant_effect = cohort[cohort.pvalue <= 0.05]
        
        sig_eff_career_ages = significant_effect.career_age.values
        sig_eff_career_ages = [index - 1 for index in sig_eff_career_ages]

        ax3[i, j].plot(cohort["career_age"], cohort["cliffd_m_f"].values, '-D', markevery=sig_eff_career_ages, linewidth=linewidth)
        ax3[i, j].plot(cohort["career_age"], [0]*cohort["career_age"].size, 'r--')

        ax3[i, j].set_ylim([-0.05, 0.22])
        ax3[i, j].set_yticks([0.0, 0.1, 0.2])
#         ax3[i, j].tick_params(labelcolor='black', top=False, bottom=True, left=True, right=True, direction='in')
        
        ax3[i, j].xaxis.set_ticks_position('bottom')
        ax3[i, j].yaxis.set_ticks_position('both')
    #     ax3[i, j].xaxis.set_minor_locator(MultipleLocator(1))
        ax3[i, j].set_title(f"{year}", fontsize=fontsize)
        ax3[i, j].set_xticks([5,10,15])
#         ax3[i, j].set_xticklabels(['', '', ''])
#         ax3[i, j].xaxis.set_major_formatter(plt.NullFormatter())

        if i == nrows-1: #last row
            plt.setp(ax3[i,j].get_xticklabels(), visible=True)
        else:
            plt.setp(ax3[i,j].get_xticklabels(), visible=False)
            
        ax3[i, j].tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize, pad=7)
        ax3[i, j].tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize, pad=7)
        ax3[i, j].tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
        ax3[i, j].tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
        ax3[i, j].spines['left'].set_linewidth(linewidth)
        ax3[i, j].spines['right'].set_linewidth(linewidth)
        ax3[i, j].spines['bottom'].set_linewidth(linewidth)
        ax3[i, j].spines['top'].set_linewidth(linewidth)
#         plt.gcf().text(0., 0.9, letter, fontsize=fontsize*2)
#         plt.subplots_adjust(left=0.25, right=0.95, bottom=0.2, top=0.9)

        if (j < cols - 1):
            j = j + 1
        else:
            j = 0
            i = i + 1

    fig3.savefig(f"./fig-7-notebook/{criterion}{ext}_gender_{num_coh}.png", facecolor=fig3.get_facecolor(), edgecolor='none',
                 bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    plt.close(fig3)


# -

# #### What are the variance differences between the population of men and women

cum_cit_pub_agg_first_m_f = cum_cit_pub_agg_first[cum_cit_pub_agg_first.gender.isin(['m', 'f'])]
cum_cit_pub_agg_first_m_f['cum_num_pub_var'] = cum_cit_pub_agg_first_m_f['cum_num_pub'].apply(np.var)
cum_cit_pub_agg_first_m_f['cum_num_cit_var'] = cum_cit_pub_agg_first_m_f['cum_num_cit'].apply(np.var)

cum_cit_pub_agg_first_m_f.groupby(['start_year', 'career_age'])['cum_num_pub_var'].apply(pd.DataFrame.diff)

# #### Plot

# +
mwu_cliff_d_cum_pub_first = get_cohort_effect_size(cum_cit_pub_agg_first, 'cum_num_pub')
mwu_cliff_d_cum_cit_first = get_cohort_effect_size(cum_cit_pub_agg_first, 'cum_num_cit')

mwu_cliff_d_cum_pub = get_cohort_effect_size(cum_cit_pub_agg, 'cum_num_pub')
mwu_cliff_d_cum_cit = get_cohort_effect_size(cum_cit_pub_agg, 'cum_num_cit')

# +
# TODO are mwu and delta same sided?
# -


def get_effect_size_stats(stats):
    effect_size_count = stats.groupby('start_year').agg({'cliffd_m_f': [lambda x: sum(x >= 0.1), lambda x: sum(x <= -0.1)],
                                                         'pvalue': lambda x: sum(x <= 0.05)})
    effect_size_count.columns = ['sig_male', 'sig_female', 'sig_mwu']
    return effect_size_count
def get_num_larger_female(stats):
    stats = stats[~(stats.start_year == 1970)]
    num_fem_larger = stats.groupby('start_year').agg({'cliffd_m_f': lambda x: sum(x < 0) })
    return num_fem_larger


def report1(stats):
    num_larg_female = get_num_larger_female(stats)
#     print(num_larg_female)
    print(f"num observations {num_larg_female.sum().values[0]}")
    print(f"num cohorts with these observations {num_larg_female[num_larg_female.cliffd_m_f > 0].shape[0]}")
report1(mwu_cliff_d_cum_cit)
report1(mwu_cliff_d_cum_cit_first)
# report1(mwu_cliff_d_cum_pub)
# report1(mwu_cliff_d_cum_pub_first)

mwu_cliff_d_cum_cit[mwu_cliff_d_cum_cit.start_year == 1991]

dfs = [mwu_cliff_d_cum_pub, mwu_cliff_d_cum_cit, mwu_cliff_d_cum_pub_first, mwu_cliff_d_cum_cit_first]
names = ['Publications', 'Citations', 'Pub. first', 'Cit. first']
res = []
res2 = []
for df in dfs:
    effect_size_count = get_effect_size_stats(df)
    effect_size_count = effect_size_count.drop(1970)
    res.append(effect_size_count[effect_size_count >= 1.0].count())
    res2.append(effect_size_count.sum())
#     print(effect_size_count)
#     print()
num_cohorts = pd.DataFrame(res)
num_cohorts = num_cohorts.T
num_cohorts.columns = names

# number of cohorts with sig_male, sig_female or sig_mwu showing up at least once 
num_cohorts

num_observations = pd.DataFrame(res2)
num_observations = num_observations.T
num_observations.columns = names

# total num of significant observations
num_observations

# +
# pubs
plot_cohort_diffs_over_ages(mwu_cliff_d_cum_pub_first, 'mwu_cliffsd_cum_pub', 'Cumulative publications first auth', ext='_first')
plot_cohort_diffs_over_ages(mwu_cliff_d_cum_pub, 'mwu_cliffsd_cum_pub', 'Cumulative publications')

# cit
plot_cohort_diffs_over_ages(mwu_cliff_d_cum_cit_first, 'mwu_cliffsd_cum_cit', 'Cumulative citations first auth', ext='_first')
plot_cohort_diffs_over_ages(mwu_cliff_d_cum_cit, 'mwu_cliffsd_cum_cit', 'Cumulative citations')

# pubs
plot_cohort_diffs_over_ages(mwu_cliff_d_cum_pub_first, 'mwu_cliffsd_cum_pub', 'Cumulative publications first auth',
                            ext='_first', remove_half=True)
plot_cohort_diffs_over_ages(mwu_cliff_d_cum_pub, 'mwu_cliffsd_cum_pub', 'Cumulative publications', remove_half=True)

# cit
plot_cohort_diffs_over_ages(mwu_cliff_d_cum_cit_first, 'mwu_cliffsd_cum_cit', 'Cumulative citations first auth', 
                            ext='_first', remove_half=True)
plot_cohort_diffs_over_ages(mwu_cliff_d_cum_cit, 'mwu_cliffsd_cum_cit', 'Cumulative citations', remove_half=True)
# -
# # Daniel, this is new code to create plots for horizontal inequality. Please reformat to make it nice like the other stuff


max([mwu_cliff_d_cum_pub.cliffd_m_f.max(), 
mwu_cliff_d_cum_pub_first.cliffd_m_f.max(), 
mwu_cliff_d_cum_cit.cliffd_m_f.max(), 
mwu_cliff_d_cum_cit_first.cliffd_m_f.max()])

# +
#df = mwu_cliff_d_cum_pub.copy()
#letter = 'A'
#title = 'Productivity'

#df = mwu_cliff_d_cum_cit.copy()
#letter = 'B'
#title = 'Recognition'

#df = mwu_cliff_d_cum_pub_first.copy()
#letter = 'C'
#title = 'Productivity'

df = mwu_cliff_d_cum_cit_first.copy()
letter = 'D'
title = 'Recognition'

df.loc[df['pvalue'] > 0.05, 'cliffd_m_f'] = 0
df = df.pivot(index='career_age', columns='start_year', values='cliffd_m_f').sort_index(ascending=False)

linewidth = 2
fontsize = 18
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
plot = ax.imshow(
    df, 
    cmap='bwr', 
    vmin=-0.17490821707496162, 
    vmax=0.17490821707496162, 
    aspect=31/15
)

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.set_xticks([0, 10, 20, 30])
ax.set_xticklabels(['1970', '1980', '1990', '2000'])
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.set_yticks([0, 5, 10, 14])
ax.set_yticklabels(['15', '10', '5', '1'])
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.set_xlabel('Cohort Start Year', fontsize=fontsize)
ax.set_ylabel('Career Age', fontsize=fontsize)
ax.set_title(title, fontsize=fontsize)
ax.tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize, pad=7)
ax.tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize, pad=7)
ax.tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
ax.tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['right'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['top'].set_linewidth(linewidth)
plt.gcf().text(0., 0.9, letter, fontsize=fontsize*2)
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)

#fig.savefig('fig-7-notebook/horiz_ineq_a.pdf')
#fig.savefig('fig-7-notebook/horiz_ineq_b.pdf')
#fig.savefig('fig-7-notebook/horiz_ineq_c.pdf')
fig.savefig('fig-7-notebook/horiz_ineq_d.pdf')
# -

from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

linewidth = 2
fontsize = 18
fig = plt.figure(figsize=(1.5, 4))
ax = fig.add_subplot(111)
col_map = plt.get_cmap('bwr')
cb = ColorbarBase(ax, cmap=col_map, orientation='vertical', norm=Normalize(-.17490821707496162, .17490821707496162))
cb.set_label('Cliff\'s $d$', fontsize=fontsize)
cb.outline.set_linewidth(linewidth)
ax.tick_params(labelsize=fontsize, width=linewidth, size=4, direction='in')
plt.subplots_adjust(left=0.1, right=0.3, bottom=0.2, top=0.9)
fig.savefig('fig-7-notebook/horiz_ineq_colorbar.pdf')

print('significant differences')
print(
    '... productivity:', 
    len(mwu_cliff_d_cum_pub[mwu_cliff_d_cum_pub['pvalue'] <= 0.05])
)
print(
    '... recognition:', 
    len(mwu_cliff_d_cum_cit[mwu_cliff_d_cum_cit['pvalue'] <= 0.05])
)
print(
    '... productivity (first):', 
    len(mwu_cliff_d_cum_pub_first[mwu_cliff_d_cum_pub_first['pvalue'] <= 0.05])
)
print(
    '... recognition (first):', 
    len(mwu_cliff_d_cum_cit_first[mwu_cliff_d_cum_cit_first['pvalue'] <= 0.05])
)
print('of', 15*31, 'observations')

# +
mwu_cliff_d_cum = pd.merge(
    left=mwu_cliff_d_cum_pub[mwu_cliff_d_cum_pub['pvalue'] <= 0.05][['start_year', 'career_age', 'cliffd_m_f']], 
    right=mwu_cliff_d_cum_cit[mwu_cliff_d_cum_cit['pvalue'] <= 0.05][['start_year', 'career_age', 'cliffd_m_f']], 
    on=['start_year', 'career_age']
)
mwu_cliff_d_cum.columns = ['start_year', 'career_age', 'cliffd_m_f_pub', 'cliffd_m_f_cit']

mwu_cliff_d_cum_first = pd.merge(
    left=mwu_cliff_d_cum_pub_first[mwu_cliff_d_cum_pub_first['pvalue'] <= 0.05][['start_year', 'career_age', 'cliffd_m_f']], 
    right=mwu_cliff_d_cum_cit_first[mwu_cliff_d_cum_cit_first['pvalue'] <= 0.05][['start_year', 'career_age', 'cliffd_m_f']], 
    on=['start_year', 'career_age']
)
mwu_cliff_d_cum_first.columns = ['start_year', 'career_age', 'cliffd_m_f_pub', 'cliffd_m_f_cit']
# -



import scipy as sp

# +
df = mwu_cliff_d_cum_first

matrix_r = []
matrix_p = []
for i in range(4):
    row_r = []
    row_p = []
    for j in range(4):
        r, p = sp.stats.pearsonr(
            df.iloc[:, i], 
            df.iloc[:, j]
        )
        row_r.append(r)
        row_p.append(p)
    matrix_r.append(row_r)
    matrix_p.append(row_p)
print('r', pd.DataFrame(matrix_r, index=df.columns, columns=df.columns))
print('p', pd.DataFrame(matrix_p, index=df.columns, columns=df.columns))
# -







# ## Inequality of papers

def plot_ineq_papers_cohort(cohort_year, years_in_future=5, career_ages=[0,1,2,5,10], func=gini,
                           data=uncited_papers_network_first_auth):
    
    uncited_papers_network_cohort = data[data['start_year'] 
                                                                          == cohort_year]
    paper_cited_list = uncited_papers_network_cohort.groupby(['year_pub', 'pub_id']).agg({'year_cit': list})
    for career_year in [cohort_year + ca for ca in career_ages]:
#         print(f"Career year {career_year}")
        paper_cited_list_year = paper_cited_list.loc[career_year]
#         print('Cited in: ', end=' ')
        for i in range(career_year,career_year+years_in_future):
            paper_cited_list_year[f'cit_in_{i}'] = paper_cited_list_year['year_cit'].apply(lambda x: sum(list(map(lambda y: 
                                                                                                                  y==i, x))))
#             print(i, end=' ')
#         print(paper_cited_list_year.columns)
        ginis = [func(paper_cited_list_year[col].astype(float).values) for col in paper_cited_list_year.columns[1:]]
#         print(ginis)
        plt.plot(ginis, label=f'{career_year-cohort_year}')
    plt.xlabel('Years after publishing')
    if func.__name__ == 'gini': 
        ylab = 'Gini in Recognition'
    else:
        ylab = func.display_name
    plt.ylabel(ylab)
    plt.legend(title='Career age when published')
    plt.title(f'Paper inequality for Cohort: {cohort_year}')
    plt.show()


# ### Authors with career len > 10

# +
# plot_ineq_papers_cohort(2000, 5, func=percentage_zeros, data=uncited_papers_network_first_auth_10)

# +
# plot_ineq_papers_cohort(2000, 5, func=gini, data=uncited_papers_network_first_auth_10)

# +
# plot_ineq_papers_cohort(2000, 5, func=gini_nonzero, data=uncited_papers_network_first_auth_10)
# -

# ### All authors

plot_ineq_papers_cohort(2000, 5, func=percentage_zeros)
plot_ineq_papers_cohort(2000, 5, func=gini_nonzero)
plot_ineq_papers_cohort(2000, 5, func=gini)

plot_ineq_papers_cohort(2000, 5, func=np.mean)

# ## Cohort Sizes

cohort_sizes = counts.groupby('start_year').agg({'author': 'nunique'})


def plot_cohort_size_over_years():
    linewidth = 2
    fontsize = 18
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.plot(cohort_sizes.index, cohort_sizes.values, linewidth=linewidth, color='black')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.set_yscale('log')
    ax.set_xlabel('Cohort', fontsize=fontsize)
    ax.set_ylabel('Number of Authors', fontsize=fontsize)
    #ax.set_title('', fontsize=fontsize)
    ax.set_xticks([1970, 1980, 1990, 2000])
    ax.tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize, pad=7)
    ax.tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize, pad=7)
    ax.tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
    ax.tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    #ax.legend(fontsize=fontsize)
    plt.gcf().text(0., 0.9, 'A', fontsize=fontsize*2)
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.2, top=0.9)
    fig.savefig('./fig-7-notebook/cohort_size.pdf')
plot_cohort_size_over_years()

# ## Make plot

ml = MultipleLocator(6)
ml.view_limits(1, 13)

# + {"code_folding": [39]}
from matplotlib.ticker import MultipleLocator, FixedLocator
def plot_criteria_over_career_ages(data, criteria, criteria_name, title, letter, x_start=1, x_end=15, legend=True, name_ext=''):
    linewidth = 2
    fontsize = 18
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    color = ['#000000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    cohort = [1970, 1975, 1980, 1985, 1990, 1995, 2000]
    for i in range(0, 7):
        df = data[data['start_year'] == cohort[i]]
        df = df[(df['career_age'] >= x_start) & (df['career_age'] <= x_end)]
        ax.plot(df['career_age'], df[criteria], linewidth=linewidth, label=cohort[i], color=color[i])
    ax.set_ylim([-0.05, 1.05])
    if 'gini' in criteria:
        ax.set_ylim([-0.05, 1.05])
    if 'hhi' in criteria:
        ax.set_ylim([-0.004, 0.074])
    if 'pzero' in criteria:
        ax.set_ylim([-0.05, 1.05])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel('Career Age', fontsize=fontsize)
    ax.set_ylabel(f'{criteria_name}', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    ax.tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize, pad=7)
    ax.tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize, pad=7)
    ax.tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
    ax.tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    if x_end == 13:
        ax.set_xlim([0.25, 13.75])
        ax.set_xticks([])
        ax.xaxis.set_major_locator(FixedLocator([1, 7, 13]))
        ax.set_xticks([1, 7, 13])
        ax.set_xticklabels(['3', '9', '15'])
    else:
        ax.set_xlim([0.25, 15.75])
        ax.set_xticks([1, 5, 10, 15])
    if legend: ax.legend(fontsize=fontsize-6)
    plt.gcf().text(0., 0.9, letter, fontsize=fontsize*2)
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.2, top=0.9)
    fig.savefig(f'./fig-7-notebook/{criteria}{name_ext}.pdf')
    
    
def plot_heatmap(data, publish_years, crit):
    data_matrix = data.groupby('career_age').agg({f'ec_cit_{start}_{end}{crit}': 'mean' for start,end in publish_years})
    data_matrix[data_matrix < 0.0001] = None
    sns.heatmap(data_matrix.T[::-1], cmap="YlGnBu")


# + {"code_folding": [0]}
def agg_data_df(citations_window, func, func_name, remove_zero=False):
    cohort_counts = citations_window.groupby(['start_year', 'career_age']).agg({
    'cum_num_pub':func, 'cum_num_cit':func, 'win_num_pub':func, 'win_num_cit':func})
    # remove people not competing for attention
    if remove_zero:
        raise Exception("DONT REMOVE ZERO!")
        cohort_counts_pub_gt0 = citations_window[citations_window['win_num_pub']>0].groupby(
            ['start_year', 'career_age']).agg({'win_num_cit':func, 'win_num_pub':func})
        cohort_counts['win_num_cit'] = cohort_counts_pub_gt0['win_num_cit']
        cohort_counts['win_num_pub'] = cohort_counts_pub_gt0['win_num_pub']
    cohort_counts.reset_index(inplace=True)
    cohort_counts = cohort_counts.rename({
        'cum_num_pub':f'{func_name}_cum_num_pub', 
        'cum_num_cit':f'{func_name}_cum_num_cit',
        'win_num_pub':f'{func_name}_win_num_pub',
        'win_num_cit': f'{func_name}_win_num_cit'
    }, axis='columns')
    return cohort_counts
# + {"code_folding": []}
def plot_array_configs(data, configs, letters, x_ends, name_ext=''):
    for config, letter, x_end in zip(configs, letters, x_ends):
        legend = False
        if letter == 'A': legend = True
        plot_criteria_over_career_ages(data, *config, letter=letter, legend=legend, x_end=x_end, name_ext=name_ext)
        
def plot_array_configs2(data, configs, letters, name_ext=''):
    for config, letter in zip(configs, letters):
        legend = False
        if letter in ['A', 'E', 'I']: legend = True
        plot_criteria_over_cohorts(data, *config, letter=letter, legend=legend, name_ext=name_ext)


# + {"code_folding": []}
def get_num_auth(citations_window):
    return citations_window['author'].nunique()

def get_config(crit, crit_name, size=""):
    config1 = [(f'{crit}_cum_num_pub', f'{crit_name}', f'Productivity {size}'), #Cumulative
               (f'{crit}_win_num_pub', f'{crit_name}', f'Productivity {size}'), #Window 
                (f'{crit}_cum_num_cit', f'{crit_name}', f'Recognition {size}'), #Cumulative
                (f'{crit}_win_num_cit', f'{crit_name}', f'Recognition {size}')] #Window 
    return config1

def get_config2(crit, crit_name, size=""):
    config2 = [('cum_num_pub', f'{crit_name}', 'Productivity'), #Cumulative
               ('num_pub', f'{crit_name}', 'Productivity'), #Window 
                ('cum_num_cit', f'{crit_name}', 'Recognition'), #Cumulative
                ('num_cit', f'{crit_name}', 'Recognition')] #Window 
    return config2

letters1 = ['A', 'B', 'C', 'D']
letters2 = ['E', 'F', 'G', 'H']
letters3 = ['I', 'J', 'K', 'L']

letters11 = ['A', 'A', 'B', 'B']
letters12 = ['C', 'C', 'D', 'D']
letters21 = ['E', 'E', 'F', 'F']
letters22 = ['G', 'G', 'H', 'H']
letters31 = ['I', 'I', 'J', 'J']
letters32 = ['K', 'K', 'L', 'L']

x_ends = [15,13,15,13]
x_ends2 = [15,11,15,11]


# + {"code_folding": [0]}
def plot_early_late_work(author_early_work, years_list, name_ext=''):
    num_iter = len(years_list)
    crits = ['_cum', '']
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    titles = ['Cum', 'Win']

    legend = True
    for i in range(0,num_iter*len(crits)):
        letter=letters[i]
        start,end = years_list[i%num_iter]
        crit = crits[i//num_iter]
        title = titles[i//num_iter]
        plot_criteria_over_career_ages(author_early_work, f'ec_cit_{start}_{end}{crit}', 'Gini Recognition', 
            title=f'{title}. (Papers of Age {start+1})', x_start=end,
            letter=letter, legend=legend, name_ext=name_ext)
        legend = False


# + {"code_folding": [0]}
def agg_data_early_late(citations_window, func, publish_years):
    aggregate = {f'ec_cit_{start}_{end}': func for start,end in publish_years}
    aggregate.update({f'ec_cit_{start}_{end}_cum': func for start,end in publish_years})

    author_early_work = citations_window[citations_window['num_pub']>0].groupby(['start_year', 'career_age']).agg(aggregate).reset_index()
    return author_early_work


# -

# ###### Plot for latex

# + {"code_folding": [1]}
# %%time
def plot_for_latex2(citations_window, citations_window_first, x_ends):
    # all authors
    cohort_counts_gini = agg_data_df(citations_window, gini, 'gini')
    plot_array_configs(cohort_counts_gini, get_config('gini', 'Gini'), letters1, x_ends)
    # zero removed
    cohort_counts_gini = agg_data_df(citations_window, gini, 'gini', remove_zero=True)
    plot_array_configs(cohort_counts_gini, get_config('gini', 'Gini'), letters2, x_ends, name_ext='_nozero')
    # zero percentage
    cohort_counts_pzero = agg_data_df(citations_window, percentage_zeros, 'pzero')
    plot_array_configs(cohort_counts_pzero, get_config('pzero', '%0'), letters3, x_ends)
    
def plot_for_latex(citations_window, citations_window_first, x_ends):
    # all authors
    cohort_counts_gini = agg_data_df(citations_window, gini, 'gini')
    plot_array_configs(cohort_counts_gini, get_config('gini', 'Gini'), letters11, x_ends)
    # first author
    cohort_counts_gini_first = agg_data_df(citations_window_first, gini, 'gini')
    plot_array_configs(cohort_counts_gini_first, get_config('gini', 'Gini'), letters12, x_ends, name_ext='_first')
    # dropouts removed 10y
    citations_window_stayed = citations_window[citations_window.dropped_after_10 == False]
    cohort_counts_stayed_gini = agg_data_df(citations_window_stayed, gini, 'gini')
    plot_array_configs(cohort_counts_stayed_gini, get_config('gini', 'Gini'), letters21, x_ends, name_ext='_stay10')
    # dropouts removed first auth 10y
    citations_window_stayed_first = citations_window_first[citations_window_first.dropped_after_10 == False]
    cohort_counts_stayed_gini_first = agg_data_df(citations_window_stayed_first, gini, 'gini')
    plot_array_configs(cohort_counts_stayed_gini_first, get_config('gini', 'Gini'), letters22, x_ends, name_ext='_stay10_first')
    # dropouts removed 5y
#     citations_window_stayed_5 = citations_window[citations_window.dropped_after_5 == False]
#     cohort_counts_stayed_gini_5 = agg_data_df(citations_window_stayed_5, gini, 'gini')
#     plot_array_configs(cohort_counts_stayed_gini_5, get_config('gini', 'Gini', get_num_auth(citations_window_stayed_5)), letters31, x_ends, name_ext='_stay5')
    # dropouts removed first auth 5y
#     citations_window_stayed_first_5 = citations_window_first[citations_window_first.dropped_after_5 == False]
#     cohort_counts_stayed_gini_first_5 = agg_data_df(citations_window_stayed_first_5, gini, 'gini')
#     plot_array_configs(cohort_counts_stayed_gini_first_5, get_config('gini', 'Gini', get_num_auth(citations_window_stayed_first_5)), letters32, x_ends, name_ext='_stay5_first')

plot_for_latex(citations_window, citations_window_first, x_ends)
# plot_for_latex(citations_window_5, citations_window_first_5, x_ends2)
# plot_for_latex2(citations_window, citations_window_first, x_ends)
# -

# ## Ginis

# #### First author

cohort_counts_gini.head()

cohort_counts_gini_first = agg_data_df(citations_window_first, gini, 'gini')

plot_array_configs(cohort_counts_gini_first, get_config('gini', 'Gini'), letters3, x_ends, name_ext='_first')

# #### Active authors

# +
act1 = citations_window[((citations_window['career_age']==1) & (citations_window['num_pub']>0))].author
act2 = citations_window[((citations_window['career_age']==2) & (citations_window['num_pub']>0))].author
act3 = citations_window[((citations_window['career_age']==3) & (citations_window['num_pub']>0))].author

act123 = list(set(act1).intersection(set(act2)).intersection(act3))
# -

citations_window_active = citations_window[citations_window.author.isin(act123)]

cohort_counts_gini_active = agg_data_df(citations_window_active, gini, 'gini')

plot_array_configs(cohort_counts_gini_active, get_config('gini', 'Gini'), letters31, x_ends, name_ext='_active')

# #### Active first authors

citations_window_active_first = citations_window_first[citations_window_first.author.isin(act123)]
cohort_counts_gini_active_first = agg_data_df(citations_window_active_first, gini, 'gini')
plot_array_configs(cohort_counts_gini_active_first, get_config('gini', 'Gini'), letters32, x_ends, name_ext='_active_first')

# #### All authors

cohort_counts_gini = agg_data_df(citations_window, gini, 'gini')

plot_array_configs(cohort_counts_gini, get_config('gini', 'Gini'), letters1, x_ends)

# #### Non zero

cohort_counts_gini_nonzero = agg_data_df(citations_window, gini_nonzero, 'gini_nonzero')

plot_array_configs(cohort_counts_gini_nonzero, get_config('gini_nonzero', 'Gini$_{>0}$'), letters1, x_ends)

# #### Inequality of early vs later work

# +
# publish_years = [[0,3], [3,6], [6,9], [0,1], [3,4], [6,7]]
# first_year = 0
# publish_years = [[i, i+1] for i in range(first_year,15)]
# author_gini_early_work = agg_data_early_late(citations_window, gini, publish_years)

# +
# plot_heatmap(author_gini_early_work, publish_years, '_cum')

# +
# plot_heatmap(author_gini_early_work, publish_years, '')

# +
# years_list = [[0,1], [3,4], [6,7]]
# plot_early_late_work(author_gini_early_work, years_list)
# -

# # Daniel, here I explore the Gini data

# files from which plots are made
gini_cohorts_ca
gini_cohorts_ca_first
gini_cohorts_ca_stayed
gini_cohorts_ca_stayed_first
gini_cohorts_ca_active
gini_cohorts_ca_active_first

gini_cohorts_ca.tail()

min([gini_cohorts_ca[gini_cohorts_ca['career_age'] >= 3].num_pub.min(), 
gini_cohorts_ca[gini_cohorts_ca['career_age'] >= 3].num_cit.min(), 
gini_cohorts_ca_first[gini_cohorts_ca_first['career_age'] >= 3].num_pub.min(), 
gini_cohorts_ca_first[gini_cohorts_ca_first['career_age'] >= 3].num_cit.min()])

max([gini_cohorts_ca[gini_cohorts_ca['career_age'] >= 3].num_pub.max(), 
gini_cohorts_ca[gini_cohorts_ca['career_age'] >= 3].num_cit.max(), 
gini_cohorts_ca_first[gini_cohorts_ca_first['career_age'] >= 3].num_pub.max(), 
gini_cohorts_ca_first[gini_cohorts_ca_first['career_age'] >= 3].num_cit.max()])

# +
#df = gini_cohorts_ca.copy()
#series = 'num_pub'
#letter = 'A'
#title = 'Productivity'

df = gini_cohorts_ca.copy()
series = 'num_cit'
letter = 'B'
title = 'Recognition'

#df = gini_cohorts_ca_first.copy()
#series = 'num_pub'
#letter = 'C'
#title = 'Productivity'

#df = gini_cohorts_ca_first.copy()
#series = 'num_cit'
#letter = 'D'
#title = 'Recognition'

if (series == 'num_pub') | (series == 'num_cit'):
    df = df[df['career_age'] >= 3]
    aspect = 31/13
else:
    aspect = 31/15
df = df.pivot(index='career_age', columns='start_year', values=series).sort_index(ascending=False)

linewidth = 2
fontsize = 18
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
plot = ax.imshow(
    df, 
    cmap='hsv', 
    vmin=0., 
    vmax=1., 
    aspect=aspect
)

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.set_xticks([0, 10, 20, 30])
ax.set_xticklabels(['1970', '1980', '1990', '2000'])
ax.xaxis.set_minor_locator(MultipleLocator(1))
if (series == 'num_pub') | (series == 'num_cit'):
    ax.set_yticks([0, 5, 10, 12])
    ax.set_yticklabels(['15', '10', '5', '3'])
else:
    ax.set_yticks([0, 5, 10, 14])
    ax.set_yticklabels(['15', '10', '5', '1'])
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.set_xlabel('Cohort Start Year', fontsize=fontsize)
ax.set_ylabel('Career Age', fontsize=fontsize)
ax.set_title(title, fontsize=fontsize)
ax.tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize, pad=7)
ax.tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize, pad=7)
ax.tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
ax.tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['right'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['top'].set_linewidth(linewidth)
plt.gcf().text(0., 0.9, letter, fontsize=fontsize*2)
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)

#fig.savefig('fig-7-notebook/vert_ineq_win_a.pdf')
fig.savefig('fig-7-notebook/vert_ineq_win_b.pdf')
#fig.savefig('fig-7-notebook/vert_ineq_win_c.pdf')
#fig.savefig('fig-7-notebook/vert_ineq_win_d.pdf')
# -
linewidth = 2
fontsize = 18
fig = plt.figure(figsize=(1.5, 4))
ax = fig.add_subplot(111)
col_map = plt.get_cmap('hsv')
cb = ColorbarBase(ax, cmap=col_map, orientation='vertical', norm=Normalize(0., 1.))
cb.set_label('Gini', fontsize=fontsize)
cb.outline.set_linewidth(linewidth)
ax.tick_params(labelsize=fontsize, width=linewidth, size=4, direction='in')
plt.subplots_adjust(left=0.1, right=0.3, bottom=0.2, top=0.9)
fig.savefig('fig-7-notebook/vert_ineq_win_colorbar.pdf')






# ## HHI

cohort_counts_hhi = agg_data_df(citations_window, hhi, 'hhi')

plot_array_configs(cohort_counts_hhi, get_config('hhi', 'HHI'), letters1, x_ends)

# ## Percentage zeros

cohort_counts_pzero = agg_data_df(citations_window, percentage_zeros, 'pzero')

plot_array_configs(cohort_counts_pzero, get_config('pzero', '%0'), letters1, x_ends)

# ## Remove dropouts

citations_window_stayed = citations_window[citations_window.dropped_after_10 == False]

citations_window_stayed_size = citations_window_stayed['author'].nunique()

# ### Gini

cohort_counts_stayed_gini = agg_data_df(citations_window_stayed, gini, 'gini')

plot_array_configs(cohort_counts_stayed_gini, get_config('gini', 'Gini', citations_window_stayed_size), 
                   letters2, x_ends, name_ext='_stay')

# ### P zero

cohort_counts_stayed_pzero = agg_data_df(citations_window_stayed, percentage_zeros, 'pzero')
plot_array_configs(cohort_counts_stayed_pzero, get_config('pzero', '%0'), letters1, x_ends)

# ### Early vs late

# +
# publish_years = [[0,3], [3,6], [6,9], [0,1], [3,4], [6,7]]
publish_years = [[i, i+1] for i in range(1,15)]

author_gini_early_work_stayed = agg_data_early_late(citations_window_stayed, gini, publish_years)
# -

plot_heatmap(author_gini_early_work_stayed, publish_years, '')

plot_heatmap(author_gini_early_work_stayed, publish_years, '_cum')

author_gini_early_work_stayed.columns

years_list = [[1,2], [3,4], [6,7]]
plot_early_late_work(author_gini_early_work_stayed, years_list, name_ext='_stay')

stop here

# ## Distributions

# ### Cohort Years
# Cumulative numbers needed.

start_year = 2000
career_age = 15
data = citations_window[(citations_window['start_year'] == start_year) & (citations_window['career_age'] == career_age)]['cum_num_pub']
data = data[data > 0]
#np.median(data)
#np.round(np.mean(data))

# pdf='cum_num_pub_2000.pdf'
cs.fit_power_law(data, discrete=True, xmin=1, xlabel='p', title='2000', bins=24, bootstrap=None, col=1, marker='o', markersize=12, linewidth=2, fontsize=24, unbinned_data=True, pdf=None, png=None)

#guys from 2000 cohort with more than 300 papers
counts[(counts['start_year'] == 2000) & (counts['career_age'] == 15) & (counts['cum_num_pub'] > 300)][['author', 'cum_num_pub']]

# ### Years

citations_window.head()

P_15 = citations_window[citations_window['career_age'] == 15]['cum_num_pub'].to_list()
C_15 = citations_window[citations_window['career_age'] == 15]['cum_num_cit'].to_list()

# publications produced between 1971 and 2014
#p = list(counts[counts['year'].between(1971, 2014)][['author', 'num_pub']].groupby('author').sum().reset_index(drop=True)['num_pub'])
# publications by all authors from cohorts 1971 to 2000 produced in their first 15 years
p = list(citations_window[(citations_window['start_year'].between(1971, 2000)) & (citations_window['career_age'].between(1, 15))][['author', 'num_pub']].groupby('author').sum().reset_index(drop=True)['num_pub'])
p = [int(x) for x in p if x>0]

# citations received between 1971 and 2014
#c = list(counts[counts['year'].between(1971, 2014)][['author', 'num_cit']].groupby('author').sum().reset_index(drop=True)['num_cit'])
# citations received by all authors from cohorts 1971 to 2000 in their first 15 years
c = list(citations_window[(citations_window['start_year'].between(1971, 2000)) & (citations_window['career_age'].between(1, 15))][['author', 'num_cit']].groupby('author').sum().reset_index(drop=True)['num_cit'])
c = [int(x) for x in c if x>0]

import compsoc as cs

parameters_p, test_statistics_p = cs.fit_power_law(l=P_15, discrete=True, xmin=1, fit=None, sims=None, bootstrap=None, data_original=False, markersize=9, linewidth=2, fontsize=18, marker=0, color=1, xlabel='p', title='', legend=False, letter='', Pdf=None, png=None)

parameters_c, test_statistics_c = cs.fit_power_law(l=c, discrete=True, xmin=1, fit=None, sims=None, bootstrap=None, data_original=False, markersize=9, linewidth=2, fontsize=18, marker=0, color=2, xlabel='c', title='', legend=False, letter='', Pdf=None, png=None)

a_bin_P_15 = cs.bin_pdf(cs.pdf(P_15))
a_bin_C_15 = cs.bin_pdf(cs.pdf(C_15))

import powerlaw as pl

f_P_15 = pl.Fit(P_15, discrete=True, xmin=1)
f_C_15 = pl.Fit(C_15, discrete=True, xmin=1)

cs.bin_pdf(cs.pdf(P_15))

space_xmin_P_15 = np.logspace(np.log10(f_P_15.xmin), np.log10(max(f_P_15.data_original)), 100)
space_xmin_C_15 = np.logspace(np.log10(f_C_15.xmin), np.log10(max(f_C_15.data_original)), 100)
scale_P_15 = f_P_15.n_tail/len(f_P_15.data_original)
scale_C_15 = f_C_15.n_tail/len(f_C_15.data_original)

a_bin_C_15

plt.plot(a_bin_C_15[:, 0], a_bin_C_15[:, 3], marker='o', color='#377eb8', ls='', markersize=markersize, label='$x=C(15)$')


fontsize = 18
linewidth = 2
markersize = 9
color_full = ['#000000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
color_pale = ['#7f7f7f', '#f18c8d', '#9bbedb', '#a6d7a4', '#cba6d1', '#ffbf7f', '#ffff99', '#d2aa93', '#fbc0df', '#cccccc']
fig = plt.figure(figsize=(4, 4))
ax1 = fig.add_subplot(111)
ax1.plot(a_bin_P_15[:, 0], a_bin_P_15[:, 3], marker='o', color='#e41a1c', ls='', markersize=markersize, label='$x=P(15)$')
ax1.plot(a_bin_C_15[:, 0], a_bin_C_15[:, 3], marker='o', color='#377eb8', ls='', markersize=markersize, label='$x=C(15)$')
ax1.plot(space_xmin_P_15, scale_P_15*f_P_15.truncated_power_law.pdf(space_xmin_P_15), color='k', ls='-', linewidth=linewidth)
ax1.plot(space_xmin_C_15, scale_C_15*f_C_15.lognormal.pdf(space_xmin_C_15), color='k', ls='--', linewidth=linewidth)
ax1.xaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('$x$', fontsize=fontsize)
ax1.set_ylabel('Probability', fontsize=fontsize)
ax1.set_xticks([1, 10, 100, 1000, 10000])
ax1.tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize, pad=7)
ax1.tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize, pad=7)
ax1.tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
ax1.tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
ax1.spines['left'].set_linewidth(linewidth)
ax1.spines['right'].set_linewidth(linewidth)
ax1.spines['bottom'].set_linewidth(linewidth)
ax1.spines['top'].set_linewidth(linewidth)
ax1.legend(fontsize=fontsize-6)
plt.gcf().text(0., 0.9, 'C', fontsize=fontsize*2)
plt.subplots_adjust(left=0.25, right=0.95, bottom=0.2, top=0.9)
#fig.savefig('./fig-7-notebook/distributions.pdf')
plt.show()

gini(citations_window[citations_window['career_age'] == 15]['cum_num_pub'].values)

gini(citations_window[citations_window['career_age'] == 15]['cum_num_cit'].values)

































































































































































































year = 2014
p = counts[counts['year'] == year]['cum_num_pub']
p = p[p > 0]
#np.median(p)
#np.round(np.mean(p))

# pdf='years_cum_num_pub_2014.pdf'
cars.fit_univariate(p, discrete=True, xmin=1, xlabel='p', title=None, bins=24, bootstrap=None, col=1, marker='o', markersize=12, linewidth=2, fontsize=18, unbinned_data=True, pdf='years_cum_num_pub_2014.pdf', png='years_cum_num_pub_2014.png')

pubids = authorPublicationData.pub_id.tolist()

df_citations = pd.read_csv('data/paper_venue_citations.csv', sep='\t')

df_citations = df_citations[df_citations.year.between(1970, 2014)]
df_citations = df_citations[df_citations.paper2.isin(pubids)]

c = df_citations.groupby(['venue2']).size()

### pdf='years_cum_num_cit_2014.pdf'
cars.fit_univariate(c, discrete=True, xmin=(1, 10), xlabel='c', title=None, bins=24, bootstrap=None, col=2, marker='o', markersize=12, linewidth=2, fontsize=18, unbinned_data=True, pdf='bradford_2014.pdf', png='bradford_2014.png')

# ## Matthew Effect

# prepare output tables
micro_stats = pd.DataFrame(columns=['t', 'dt', 'author', 'eta'])
macro_stats = pd.DataFrame(columns=['t', 'dt', 'n', 'D', 'beta', 'beta_std', 'r2', 'reduced_chi'])
# settings
dt = 5
unit = 'num_cit' # CHANGE COLOR (1) AND LETTER (5)
fit = 'odr'
for t in range(1970+dt, 2015-dt+2): # range(1975, 2012) for dt=5
    # get vectors
    t0 = counts[counts['year'].between(t-dt, t-1)][['author', unit]].groupby('author').sum()
    t1 = counts[counts['year'].between(t, t+dt-1)][['author', unit]].groupby('author').sum()
    #t1 = counts[counts['year'].between(t, t)][['author', unit]].groupby('author').sum()
    t0 = t0[t0[unit] > 0]
    t1 = t1[t1[unit] > 0]
    # merge vectors into dataframe
    ca = pd.merge(left=t0, right=t1, left_index=True, right_index=True)
    ca.columns = ['t0', 't1']
    # fit
    df_stats, df_bootstrap = cars.fit_bivariate(ca['t0'], ca['t1'], fit=fit, reduction='bin', color=2, xlabel='c_{%.0f-%.0f}' %(t-dt, t-1), ylabel='c_{%.0f-%.0f}' %(t, t+dt-1), title='', pdf='matthew/macro_c_'+fit+'Bin_t'+str(t)+'_dt'+str(dt)+'.pdf')
    # extend micro stats (fitness)
    ca['t'] = t
    ca['dt'] = dt
    ca['author'] = ca.index
    t1_exp = df_stats['D'][0]*ca['t0']**df_stats['beta'][0]
    ca['eta'] = ca['t1']/t1_exp
    ca = ca[['t', 'dt', 'author', 'eta']]
    # extend macro stats
    df_stats['t'] = t
    df_stats['dt'] = dt
    df_stats = df_stats[['t', 'dt', 'n', 'D', 'beta', 'beta_std', 'r2', 'reduced_chi']]
    # append stats
    micro_stats = pd.concat([micro_stats, ca], axis=0, ignore_index=True)
    macro_stats = pd.concat([macro_stats, df_stats], axis=0, ignore_index=True)
# write output tables
micro_stats.to_csv('matthew/micro_c_'+fit+'Bin_dt'+str(dt)+'.txt', sep='\t', index=False)
macro_stats.to_csv('matthew/macro_c_'+fit+'Bin_dt'+str(dt)+'.txt', sep='\t', index=False)

# ### Cumulative Advantage

macro_p_olsBin_dt5 = pd.read_csv('matthew/macro_p_olsBin_dt5.txt', sep='\t')
macro_c_olsBin_dt5 = pd.read_csv('matthew/macro_c_olsBin_dt5.txt', sep='\t')

macro_p_odrBin_dt5 = pd.read_csv('matthew/macro_p_odrBin_dt5.txt', sep='\t')
macro_c_odrBin_dt5 = pd.read_csv('matthew/macro_c_odrBin_dt5.txt', sep='\t')

plt.fill_between(macro_p_olsBin_dt5['t'], macro_p_olsBin_dt5['beta']-macro_p_olsBin_dt5['beta_std'], macro_p_olsBin_dt5['beta']+macro_p_olsBin_dt5['beta_std'], color=color_pale[1], linewidth=0)
plt.fill_between(macro_c_olsBin_dt5['t'], macro_c_olsBin_dt5['beta']-macro_c_olsBin_dt5['beta_std'], macro_c_olsBin_dt5['beta']+macro_p_olsBin_dt5['beta_std'], color=color_pale[2], linewidth=0)
plt.plot(macro_p_olsBin_dt5['t'], macro_p_olsBin_dt5['beta'], color=color_full[1], label='Productivity')
plt.plot(macro_c_olsBin_dt5['t'], macro_c_olsBin_dt5['beta'], color=color_full[2], label='Recognition')
plt.title('Ordinary Least Squares')
plt.xlabel('Year')
plt.ylabel('Cumulative Advantage')
plt.legend()

plt.fill_between(macro_p_odrBin_dt5['t'], macro_p_odrBin_dt5['beta']-macro_p_odrBin_dt5['beta_std'], macro_p_odrBin_dt5['beta']+macro_p_odrBin_dt5['beta_std'], color=color_pale[1], linewidth=0)
plt.fill_between(macro_c_odrBin_dt5['t'], macro_c_odrBin_dt5['beta']-macro_c_odrBin_dt5['beta_std'], macro_c_odrBin_dt5['beta']+macro_p_odrBin_dt5['beta_std'], color=color_pale[2], linewidth=0)
plt.plot(macro_p_odrBin_dt5['t'], macro_p_odrBin_dt5['beta'], color=color_full[1], label='Productivity')
plt.plot(macro_c_odrBin_dt5['t'], macro_c_odrBin_dt5['beta'], color=color_full[2], label='Recognition')
plt.title('Orthogonal Distance Regression')
plt.xlabel('Year')
plt.ylabel('Cumulative Advantage')
plt.legend()

# ### Fitness

micro_p_olsBin_dt5 = pd.read_csv('matthew/micro_p_olsBin_dt5.txt', sep='\t')
micro_c_olsBin_dt5 = pd.read_csv('matthew/micro_c_olsBin_dt5.txt', sep='\t')

micro_p_odrBin_dt5 = pd.read_csv('matthew/micro_p_odrBin_dt5.txt', sep='\t')
micro_c_odrBin_dt5 = pd.read_csv('matthew/micro_c_odrBin_dt5.txt', sep='\t')

# Users with long careers:

micro_p_olsBin_dt5.groupby('author').count()

#author = 'a min tjoa'
#author = 'a-nasser ansari'
#author = 'a-xing zhu'
author = 'a. a. (louis) beex'
#author = 'a. a. agboola'
#author = 'a. a. ball'
#author = 'a. a. el-bary'
#author = 'ülkü gürler'
#author = 'ülle kotta'
#author = 'ülo nurges'
#author = 'ümit aygölü'
#author = 'ümit bilge'
#author = 'ümit güz'
#author = 'ümit v. çatalyürek'
#author = 'ümit y. ogras'
#author = 'ümit özgüner'
#author = 'ünal göktas'
#author = 'ünal ufuktepe'
p = micro_p_olsBin_dt5[micro_p_olsBin_dt5['author'] == author]
c = micro_c_olsBin_dt5[micro_c_olsBin_dt5['author'] == author]
plt.plot(p['t'], p['eta'], label='Productivity')
plt.plot(c['t'], c['eta'], label='Recognition')
plt.xlabel('Year')
plt.ylabel('Fitness')
plt.legend()

# Lognormal fitness distributions:

etaMean_p_olsBin_dt5 = micro_p_olsBin_dt5.groupby('author').mean().sort_values('eta', ascending=False)['eta']
etaMean_c_olsBin_dt5 = micro_c_olsBin_dt5.groupby('author').mean().sort_values('eta', ascending=False)['eta']

print('mean:', np.mean(np.log10(etaMean_p_olsBin_dt5)))
print('std:', np.std(np.log10(etaMean_p_olsBin_dt5)))
plt.hist(np.log10(etaMean_p_olsBin_dt5))

print('mean:', np.mean(np.log10(etaMean_c_olsBin_dt5)))
print('std:', np.std(np.log10(etaMean_c_olsBin_dt5)))
plt.hist(np.log10(etaMean_c_olsBin_dt5))

# Correlation of productivity and recognition fitness:

etaMean_olsBin_dt5 = pd.concat([etaMean_p_olsBin_dt5, etaMean_c_olsBin_dt5], axis=1)
etaMean_olsBin_dt5.columns = ['p', 'c']

etaMean_olsBin_dt5.dropna(inplace=True)

print('corrcoef:', np.corrcoef(etaMean_olsBin_dt5['p'], etaMean_olsBin_dt5['c'])[0, 1])
plt.scatter(etaMean_olsBin_dt5['p'], etaMean_olsBin_dt5['c'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Mean Productivity Fitness')
plt.ylabel('Mean Recognition Fitness')

# ## Variance of Fitness for Cohorts

counts[counts['author'] == 'a. breton']

micro_p_olsBin_dt5[micro_p_olsBin_dt5['author'] == 'a. breton']

counts_fitness_p = pd.merge(left=counts, right=micro_p_olsBin_dt5, left_on=['author', 'year'], right_on=['author', 't'])
counts_fitness_c = pd.merge(left=counts, right=micro_c_olsBin_dt5, left_on=['author', 'year'], right_on=['author', 't'])

counts_fitness_p.head()

counts_fitness_c.head()

# Note: eta for 1975 is based on productivity in 1975-1979.

counts_fitness_p_mean = counts_fitness_p.groupby(['start_year', 'career_age']).mean().reset_index()
for start_year in [1975, 1980, 1985, 1990, 1995, 2000]: # range(1975, 2012)
    df = counts_fitness_p_mean[counts_fitness_p_mean['start_year'] == start_year]
    plt.plot(df['career_age'], df['eta'], label=start_year)
    plt.xlabel('Career Age')
    plt.ylabel('Mean Productivity Fitness')
    plt.legend()

counts_fitness_c_mean = counts_fitness_c.groupby(['start_year', 'career_age']).mean().reset_index()
for start_year in [1975, 1980, 1985, 1990, 1995, 2000]: # range(1975, 2012)
    df = counts_fitness_c_mean[counts_fitness_c_mean['start_year'] == start_year]
    plt.plot(df['career_age'], df['eta'], label=start_year)
    plt.xlabel('Career Age')
    plt.ylabel('Mean Recognition Fitness')
    plt.legend()

# On average, scholars have high fitness at the beginning of their career. To which extent is the mean influenced by scholars with long careers? In other wordw, is high fitness in early years a predictor of career duration?

AB HIER BAUSTELLE

career_duration = [6, 10]
counts_fitness_p_mean = counts_fitness_p[counts_fitness_p['career_duration'].between(career_duration[0], career_duration[1], inclusive=True)].groupby('career_age').mean().reset_index()
max(counts_fitness_p_mean['career_age'])

for career_duration in [[1, 5], [6, 10], [11, 15], [15, 20], [21, 25]]:
    counts_fitness_p_mean = counts_fitness_p[counts_fitness_p['career_duration'].between(career_duration[0], career_duration[1], inclusive=True)].groupby('career_age').mean().reset_index()
    plt.plot(counts_fitness_p_mean['career_age'], counts_fitness_p_mean['eta'], label=career_duration)
    plt.xlabel('Career Age')
    plt.ylabel('Mean Productivity Fitness')
    plt.legend()

career_duration = [6, 10]
counts_fitness_c_mean = counts_fitness_c[counts_fitness_c['career_duration'].between(career_duration[0], career_duration[1], inclusive=True)].groupby('career_age').mean().reset_index()
max(counts_fitness_c_mean['career_age'])

counts_fitness_c

for career_duration in [[1, 5], [6, 10], [11, 15], [15, 20], [21, 25]]:
    counts_fitness_c_mean = counts_fitness_c[counts_fitness_c['career_duration'].between(career_duration[0], career_duration[1], inclusive=True)].groupby('career_age').mean().reset_index()
    plt.plot(counts_fitness_c_mean['career_age'], counts_fitness_c_mean['eta'], label=career_duration)
    plt.xlabel('Career Age')
    plt.ylabel('Mean Recognition Fitness')
    plt.legend()
