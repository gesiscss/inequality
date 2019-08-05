# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"pycharm": {"is_executing": false}}
import pandas as pd
import numpy as np
import calculate
from calculate import gini

# +
# Specify career len to export file for
# CAREER_LENGTH = 15
# Specify how long is the early career. Impacts which papers we take into account for early productivity and quality
# EARLY_CAREER_LEN = 3
# EARLY_CAREER_LEN_LIST = [1, 2, 3, 4, 5]
EARLY_CAREER_LEN_LIST = [3,5,7,9,11,12]
# For early career work, when do we stop counting citations. Impacts recognition
# RECOGNITION_CUT_OFF = 5
# RECOGNITION_CUT_OFF_LIST = [3, 4, 5, 6, 7, 8, 9]
# RECOGNITION_CUT_OFF_LIST = [3, 5]
RECOGNITION_CUT_OFF_LIST = [3,5,7,9,11,12]
# Success after 15 years. Impacts when we stop counting citations
SUCCESS_CUTOFF = 15
# Length of observed career for dropouts
# (1-3), middle career (4-9), late career (10-15)

# TODO: for multiple dropout intervals code does not work!!!
# CAREER_LENGTH_DROPOUTS_LIST = [ (0, 15), (0, 3), (4, 9), (10, 15)] #,
CAREER_LENGTH_DROPOUTS_LIST = [(0, 15)]
# CAREER_LENGTH_DROPOUTS = 15
INACTIVE_TIME_DROPOUTS = 10

# Specify the first and last year we consider in our analysis
START_YEAR = 1970
LAST_START_YEAR = 2000

# +
# assert(INACTIVE_TIME_DROPOUTS < CAREER_LENGTH_DROPOUTS), "Time observed for dropouts has to be smaller than the whole window!"

# +
# assert(CAREER_LENGTH >= EARLY_CAREER_LEN), "Early career len too long"
# -

# ## 1. Load data

authorPublicationData = pd.read_csv('./data/author_publications_2017_asiansAsNone.txt')
arxiv_pubid = pd.read_csv('derived-data/arxiv_pubid_2017.csv', header=None, names=['pub_id'])
authorPublicationData.head()

print(authorPublicationData.shape)
# same as dropping author, pub_id and year
authorPublicationData.drop_duplicates(subset=['author','pub_id'], inplace=True)
print(authorPublicationData.shape)
authorPublicationData = authorPublicationData.loc[~authorPublicationData.pub_id.isin(arxiv_pubid['pub_id'])]
print(authorPublicationData.shape)

authorPublicationData['pub_id'].nunique()

authorCitationsData = pd.read_csv('./data/citations_2017_asiansAsNone.txt')
authorCitationsData.head()

print(authorCitationsData.shape)
authorCitationsData.drop_duplicates(inplace=True)
print(authorCitationsData.shape)
authorCitationsData = authorCitationsData.loc[(~authorCitationsData.id1.isin(arxiv_pubid['pub_id']))| 
                                             (~authorCitationsData.id2.isin(arxiv_pubid['pub_id']))]
print(authorCitationsData.shape)

print('Authors#      - ',authorPublicationData['author'].nunique())
print('Years#        - ',authorPublicationData['year'].nunique())
print('Publications# - ',authorPublicationData['pub_id'].nunique())

# + {"pycharm": {"is_executing": false}}
# venue data
publication_venues_rank = pd.read_csv('derived-data/publication-venues-rank.csv')
# -

publication_venues_rank.head()

# ## 2. Career length and dropouts

# +
groupByAuthor = authorPublicationData.groupby(['author'])

groupByAuthorMinYearData = groupByAuthor['year'].min()
groupByAuthorMaxYearData = groupByAuthor['year'].max()
groupByAuthorCountPublicationsData = groupByAuthor['pub_id'].count()

# +
authorGroupedData = groupByAuthorMinYearData.to_frame(name='start_year')
authorGroupedData['end_year'] = groupByAuthorMaxYearData
authorGroupedData['total_num_pub'] = groupByAuthorCountPublicationsData
authorGroupedData = authorGroupedData.reset_index()
print('Total rows -                ', authorGroupedData.shape)

# authorGroupedData = authorGroupedData[authorGroupedData["start_year"] >= START_YEAR]
# print('After removing all < 1970 - ', authorGroupedData.shape)

authorGroupedData = authorGroupedData.drop_duplicates()
print('After removing duplicates - ', authorGroupedData.shape)

authorGroupedData = authorGroupedData.dropna(how='any')
print("After droping na -          ", authorGroupedData.shape)

authorGroupedData.head()
# -

# Adding 1 here to have career length be at least one. So 3 years career means year1, year2, year3.
authorGroupedData["career_length"] = authorGroupedData['end_year'] - authorGroupedData['start_year'] + 1

credible_authors = authorGroupedData

# ### Label authors that drop out

print(f"Label authors with {INACTIVE_TIME_DROPOUTS} years inacitivity in a {CAREER_LENGTH_DROPOUTS_LIST} years window as dropouts")

combined_pubs = authorPublicationData.merge(credible_authors[['author', 'start_year']], on='author', how='inner')
# TODO remove this, its the same as early career publications
print(combined_pubs.shape)


def list_append(lst, item):
    lst.append(item)
    return lst


# +
# # %%time
# This code is potato...
for start, end in CAREER_LENGTH_DROPOUTS_LIST:
    combined_pubs_grouped = combined_pubs[(combined_pubs.year >= combined_pubs.start_year + start) &
                                    (combined_pubs.year <= combined_pubs.start_year + end)]   
    # for every 2 consecutive years the author has published (nxt and prev) find a difference (absence time)
    # we artificially add two value: career start + 15 and career start, as limiters of our observation window
    # this will add 0 values in the begining for the first year
    combined_pubs_grouped = combined_pubs_grouped.groupby('author')['year', 'start_year'].apply(
        lambda x: [nxt - prev for prev, nxt in zip(sorted(list_append(list(x['year']),x['start_year'].iloc[0]+start)), 
                                                   sorted(list_append(list(x['year']),x['start_year'].iloc[0]+  end)))])
    combined_pubs_grouped = combined_pubs_grouped.reset_index()
    combined_pubs_grouped.rename({0:f'absence_list-{start}-{end}'}, inplace=True, axis='columns')
    combined_pubs_grouped[f'max_absence-{start}-{end}'] = combined_pubs_grouped[f'absence_list-{start}-{end}'].apply(max)
    combined_pubs_grouped[f'avg_absence-{start}-{end}'] = combined_pubs_grouped[f'absence_list-{start}-{end}'].apply(np.mean)
    
    credible_authors = credible_authors.merge(combined_pubs_grouped[['author', f'max_absence-{start}-{end}', 
                                                                     f'avg_absence-{start}-{end}']], on='author', how='left')
    credible_authors[f'max_absence-{start}-{end}'] = credible_authors[f'max_absence-{start}-{end}'].fillna(end-start+1)
    credible_authors[f'avg_absence-{start}-{end}'] = credible_authors[f'avg_absence-{start}-{end}'].fillna(end-start+1)
    
    # TODO: Should i also add the start year into the calculation? Now i only have end year included
# -

credible_authors['dropped_after_10'] = credible_authors['max_absence-0-15'].apply(lambda x: False if x < 10 else True)

credible_authors['max_absence-0-15'].value_counts(dropna=False)

credible_authors.shape

credible_authors['dropped_after_10'].value_counts()

credible_authors.columns

# ### Gender

# +
gender = pd.read_csv('./data/name_gender_2017_asiansAsNone_nodup.txt')
credible_authors = credible_authors.merge(gender, left_on='author', right_on='name', how='left')
credible_authors.drop('name', axis=1, inplace=True)

credible_authors.gender.value_counts()
# -

credible_authors.gender.value_counts()

gender.head()

# ### Save filtered data about authors, and cleaned publications

credible_authors[credible_authors.start_year >= START_YEAR].to_csv('derived-data/authors-scientific.csv', index=False, encoding='utf-8')
credible_authors.head()

authorPublicationData.to_csv('derived-data/author-publications.csv', index=False)

authorPublicationData.shape

# ## 3. Generate a new citation network

# ### Generate Author->Paper network

# We need data about how many times an author has been cited
# For every authors publication, i merge all citations
# Doesnt contain uncited papers
final_citation_count_from_ids = authorPublicationData.merge(authorCitationsData, left_on='pub_id', 
                                                            right_on='id2', how='inner', suffixes=('_pub', '_cit'))

print(final_citation_count_from_ids.shape)

final_citation_count_from_ids.drop_duplicates(inplace=True)

print(final_citation_count_from_ids.shape)

# #### Remove errors in citation data (years published vs years cited)

# Published before cited - NORMAL
print(final_citation_count_from_ids.shape)
num_normal = final_citation_count_from_ids[final_citation_count_from_ids.year_pub <= final_citation_count_from_ids.year_cit].shape
print(num_normal)

# Published after cited - WRONG
num_wrong = final_citation_count_from_ids[final_citation_count_from_ids.year_pub > final_citation_count_from_ids.year_cit].shape
print(num_wrong)

print("Percentage of citations to be removed: ", num_wrong[0]*100/(num_normal[0]+num_wrong[0]))
print("Less than one percent")

cit_wrong_df = final_citation_count_from_ids[final_citation_count_from_ids.year_pub > final_citation_count_from_ids.year_cit]

cit_wrong = final_citation_count_from_ids[final_citation_count_from_ids.year_pub > final_citation_count_from_ids.year_cit].index

# +
final_citation_count_from_ids.drop(cit_wrong, inplace=True)

assert num_normal[0] == final_citation_count_from_ids.shape[0], "The number of citations doesnt match"
# -

final_citation_count_from_ids.columns

# #### Save

# +
# final_citation_count_from_ids.to_csv('./data/authors_cited_by_papers_2017_asiansAsNone_by_daniel.txt',
#                                      columns=['author', 'year_pub', 'pub_id', 'id1', 'year_cit'], index=False)

final_citation_count_from_ids[['author', 'id1', 'id2', 'year_cit']].drop_duplicates().to_csv('derived-data/author-paper-citations-cleaned.csv', 
                                                                                   index=False)

# final_citation_count_from_ids.drop_duplicates(subset=['author_cited', 'pub_id_cited', 'pub_id_citing', 'author_citing'],
#                                               inplace=True)
# -

# drop duplicates on id1,id2 because we only care about paper->paper citations
paper_citation_count = final_citation_count_from_ids.drop_duplicates(subset=['id1', 'id2']).groupby('id2')['id1'].count()
paper_citation_count.to_csv('derived-data/paper-citation-count.csv')

# Its important to keep using this file for citations. As it has bad entries removed

# ### Group citations over authors and years

citations_year_auth = final_citation_count_from_ids.groupby(['author', 'year_cit'])['id1'].count()


citations_year_auth.head()

# +
citations_year_auth = citations_year_auth.reset_index()
citations_year_auth = citations_year_auth.rename(columns={'id1':'cit_count'})

citations_year_auth[['author', 'year_cit', 'cit_count']].to_csv('derived-data/authors-perYear-citations.csv', index=False)
# -

citations_year_auth = citations_year_auth.groupby(['author', 'year_cit'])['cit_count'].sum()
citations_year_auth = citations_year_auth.reset_index()

# ## Early career analysis

# Doesnt contain uncited papers
# Does not contain multiple authors per one citation
combined = final_citation_count_from_ids.merge(credible_authors[['author', 'start_year']], on='author', how='inner')
# TODO Is this 'inner' here good?

early_career_publications = authorPublicationData.merge(credible_authors[['author', 'start_year']], on='author', how='left')

print(early_career_publications.author.nunique())
print(early_career_publications.pub_id.nunique())
print(early_career_publications.shape[0])

print(combined.author.nunique())
print(combined.pub_id.nunique())
print(combined.shape[0])

combined.head()

combined.drop_duplicates(subset=['author', 'id1', 'id2'], inplace=True)

combined.columns

from memory import show_mem_usage

show_mem_usage()

# ### Author order
# The size of this df is about 3M, corresponding to the number of unique `pub_id` we have in the dataset

author_order = pd.read_csv('derived-data/publication_authors_order_2017.csv')
print(author_order.columns)
print(author_order.shape)

author_order = author_order.merge(credible_authors[['author', 'start_year']], left_on='first_author', 
                                  right_on='author', how='left')
author_order = author_order.drop('first_author', axis='columns')

for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    author_order_early = author_order[(author_order.year < author_order.start_year + EARLY_CAREER)]
    author_order_early = author_order_early.groupby('author').agg({'pub_id':'count'}).reset_index()
    author_order_early.rename({'pub_id': f'ec_first_auth_{EARLY_CAREER}'}, axis='columns', inplace=True)
    credible_authors = credible_authors.merge(author_order_early, on='author', how='left')
    credible_authors[f'ec_first_auth_{EARLY_CAREER}'] = credible_authors[f'ec_first_auth_{EARLY_CAREER}'].fillna(0)



# ### Team size

for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    early_career_publications_filtered = early_career_publications[(
        early_career_publications.year < early_career_publications.start_year + EARLY_CAREER)]
    paper_team_size = early_career_publications_filtered.groupby('pub_id').agg({'author' : 'nunique'}).reset_index()
    paper_team_size = paper_team_size.rename({'author': f'team_size_{EARLY_CAREER}'}, axis='columns')
    early_career_publications_filtered = early_career_publications_filtered.merge(paper_team_size, on='pub_id', how='left')
    team_size_median = early_career_publications_filtered.groupby('author').agg({f'team_size_{EARLY_CAREER}': 'median'}).reset_index()
    team_size_median = team_size_median.rename({f'team_size_{EARLY_CAREER}': f'team_size_median_{EARLY_CAREER}'}, axis='columns')
    team_size_mean = early_career_publications_filtered.groupby('author').agg({f'team_size_{EARLY_CAREER}': 'mean'}).reset_index()
    team_size_mean = team_size_mean.rename({f'team_size_{EARLY_CAREER}': f'team_size_mean_{EARLY_CAREER}'}, axis='columns')

    credible_authors = credible_authors.merge(team_size_median, on='author', how='left')
    credible_authors = credible_authors.merge(team_size_mean, on='author', how='left')

early_career_publications_filtered.shape

team_size_median.shape

credible_authors.columns

# #### Per Year

# +
# this has no effect, arxiv removed beforehand
# print(authorPublicationData.shape)
# print(authorPublicationData.loc[~authorPublicationData.pub_id.isin(arxiv_pubid['pub_id'])].shape)
# -

team_size = authorPublicationData.groupby(['year', 'pub_id']).size().reset_index(name='team_size')
team_size.head()

team_size_year = team_size.groupby('year').agg({'team_size': 'mean'}).reset_index()
team_size_year.head()

import matplotlib.pyplot as plt
plt.plot(team_size_year['year'], team_size_year['team_size'])
plt.yscale('log')
plt.savefig('./fig/average_team_size_across_years.png')



team_size_year.to_csv('derived-data/team_size_avg_per_year.csv', index=None)

# Are all authors included here? Also those without gender?

# ### Rolling citations

author_year_numPub = authorPublicationData.groupby(['author', 'year'])['pub_id'].count().reset_index()
author_year_numPub = author_year_numPub.rename(columns={'pub_id':'num_pub'})

# + {"pycharm": {"is_executing": false}}
all_years = credible_authors.start_year.unique()
start_years = [year for year in all_years if START_YEAR <= year <= LAST_START_YEAR]
start_years = sorted(start_years)
# -

counts0 = credible_authors[['author', 'start_year']].copy()
#filter out start years
counts0 = counts0[counts0['start_year'].isin(start_years)]
counts0['year'] = counts0['start_year'].apply(lambda x: [x+i for i in range(0, 15)])
counts = pd.DataFrame(counts0['year'].tolist(), index=counts0['author']).stack().reset_index(
    level=1, drop=True).reset_index(name='year')[['year','author']]
counts = counts.merge(credible_authors[['author', 'start_year', 'end_year', 'gender']], on='author', how='inner')
counts['career_age'] = counts['year'] - counts['start_year'] + 1
counts['year'] = counts['year'].astype('int32')

#citations window
WINDOW_SIZE = 3
df_list = []
for year in start_years:
    df_year = combined[combined.start_year == year]
    for y in range(year, year+13): #y is the first year we count for
        df_window = df_year[(df_year.year_pub>=y) & (df_year.year_cit>=y)& (df_year.year_pub<=y)& (df_year.year_cit<=y+WINDOW_SIZE)]
        df_window = df_window.groupby('author').agg({'id1': 'count'}).reset_index()
        df_window['year'] = y
        df_window = df_window.rename({'id1' : f'cit_{WINDOW_SIZE}'}, axis=1)
        df_list.append(df_window)
df_cit_3_window = pd.concat(df_list).sort_values(by=['author', 'year'])

counts = counts.merge(df_cit_3_window, on=['author', 'year'], how='left')
counts['cit_3'] = counts['cit_3'].fillna(0)

citations_year_auth.rename(columns={'year_cit':'year', 'cit_count':'num_cit'}, inplace=True)

# merge in publications
counts = counts.merge(author_year_numPub, on=['author', 'year'], how='left')
counts['num_pub'] = counts['num_pub'].fillna(0)
# merge in citations
counts = counts.merge(citations_year_auth, on=['author', 'year'], how='left')
counts['num_cit'] = counts['num_cit'].fillna(0)

counts = calculate.calculate_cumulative_for_authors(counts, 'num_cit')
counts = calculate.calculate_cumulative_for_authors(counts, 'num_pub')

counts['career_duration'] = counts['end_year']-counts['start_year'] + 1

# # %%time
#publication window
# TODO sort?!
counts['win_num_pub'] = counts.groupby('author')['num_pub'].transform(lambda x: x.rolling(3, min_periods=3).sum().shift(-2))

counts.columns

# +
# # %%time
# #unpack list of values
# df = df['year_cit'].apply(pd.Series) \
#     .join(df) \
#     .drop(["year_cit"], axis = 1) \
#     .melt(id_vars = ['author', 'year_pub'], value_name = "year_cit") \
#     .drop("variable", axis = 1) \
#     .dropna()
# -

# ### Early, mid and late papers analysis - citations

# # %%time
# publish_years = [[0,3], [3,6], [6,9], [0,1], [3,4], [6,7]]
publish_years = [[i, i+1] for i in range(0,15)]
for start,end in publish_years:
    first_3 = combined[(combined.year_pub >= combined.start_year + start) & (combined.year_pub < combined.start_year + end)]
    first_3 = first_3.groupby(['author', 'year_cit']).agg({'id1': 'count'}).reset_index()
    first_3 = first_3.rename({'year_cit' : 'year', 'id1': f'ec_cit_{start}_{end}'}, axis=1)
    counts = counts.merge(first_3, on=['author', 'year'], how='left')
    counts[f'ec_cit_{start}_{end}'] = counts[f'ec_cit_{start}_{end}'].fillna(0)

# calculate cumulative out of absolute
for start,end in publish_years:
    counts[f'ec_cit_{start}_{end}_cum'] = counts.sort_values(['author', 'career_age']).groupby('author')[f'ec_cit_{start}_{end}'].transform(pd.Series.cumsum)

# calc gini for absolute => 7
author_gini_early_work = counts.groupby(['start_year', 'career_age']).agg(
    {f'ec_cit_{start}_{end}': gini for start,end in publish_years}).reset_index()

author_gini_early_work.head()

# calc gini for cumulative => 7
author_gini_early_work_cum = counts.groupby(['start_year', 'career_age']).agg(
    {f'ec_cit_{start}_{end}_cum': gini for start,end in publish_years}).reset_index()

author_gini_early_work_cum.head()

counts.columns

counts.to_csv(f'derived-data/citations_window_{WINDOW_SIZE}.csv', index=None)

# ### Venues

early_career_venues = early_career_publications.merge(publication_venues_rank[[
    'pub_id', 'h5_index', 'ranking', 'deciles', 'quantiles']], on='pub_id', how='inner')

early_career_venues.author.nunique()

EARLY_CAREER_LEN_LIST


# +
# TODO including the MAX and MIN values as missing. Check this. also what to do with ranking?
def quantile_binary(quant): 
    return quant <= 3

for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
#     EARLY_CAREER = 3
    early_career_venues_ec = early_career_venues[early_career_venues.year < early_career_venues.start_year + EARLY_CAREER]
    early_career_venues_gr = early_career_venues_ec.groupby('author').agg({
        'h5_index': 'max',
    #     'ranking': 'min',
        'deciles': 'min',
        'quantiles': 'min'}).rename(columns={
        'h5_index': f'h5_index_max_{EARLY_CAREER}', 
    #     'ranking': f'ranking_{EARLY_CAREER}',
        'deciles': f'deciles_min_{EARLY_CAREER}',
        'quantiles': f'quantiles_min_{EARLY_CAREER}'})
    early_career_venues_gr = early_career_venues_gr.reset_index()
    credible_authors = credible_authors.merge(early_career_venues_gr, on='author', how='left')

    credible_authors[f'h5_index_max_{EARLY_CAREER}'] = credible_authors[f'h5_index_max_{EARLY_CAREER}'].fillna(0)
    credible_authors[f'deciles_min_{EARLY_CAREER}'] = credible_authors[f'deciles_min_{EARLY_CAREER}'].fillna(10)
    credible_authors[f'quantiles_min_{EARLY_CAREER}'] = credible_authors[f'quantiles_min_{EARLY_CAREER}'].fillna(4)
    
    # CLAUDIA this should be TRUE if the author has AT LEAST one paper that exceeds the threshold
    credible_authors[f'quantiles_bin_{EARLY_CAREER}'] = credible_authors[f'quantiles_min_{EARLY_CAREER}'].apply(quantile_binary)
    
    # credible_authors[f'ranking_{EARLY_CAREER}'] = credible_authors[f'ranking_{EARLY_CAREER}'].fillna(0)
# -

# ### Early degree

# +
# TODO: Can this be based on combined? Do we loose some info here
# -

for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    combined_early_degree = combined[(combined.year_pub < combined.start_year + EARLY_CAREER)]

    combined_early_degree = combined_early_degree.drop_duplicates(subset=['author', 'pub_id'])

    combined_early_degree = combined_early_degree[['author', 'pub_id']]

    # authors_per_paper = combined_early_degree.groupby('pub_id')['author'].count().reset_index()
    # authors_per_paper.rename({"author":"early_career_degree"}, axis='columns', inplace=True)

    combined_early_degree = combined_early_degree.merge(combined, on='pub_id')

    combined_early_degree = combined_early_degree[combined_early_degree.author_x != combined_early_degree.author_y]
    combined_early_degree = combined_early_degree.drop_duplicates(subset=['author_x', 'author_y'])

    combined_early_degree = combined_early_degree.groupby('author_x')['author_y'].count().reset_index()

    combined_early_degree.rename({"author_x":"author", "author_y": f"early_career_degree_{EARLY_CAREER}"}, 
                                 axis='columns', inplace=True)

    credible_authors = credible_authors.merge(combined_early_degree, on='author', how='left')
    credible_authors[f"early_career_degree_{EARLY_CAREER}"] = credible_authors[f"early_career_degree_{EARLY_CAREER}"].fillna(0)

combined_early_degree.sample(10)

# ### Early quality

for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    combined_early_quality = combined[(combined.year_pub < combined.start_year + EARLY_CAREER) &
             (combined.year_cit < combined.start_year + SUCCESS_CUTOFF)]
    
    author_order_early = author_order[(author_order.year < author_order.start_year + EARLY_CAREER)]
    early_career_quality_first = combined_early_quality.loc[combined_early_quality['pub_id'].isin(author_order_early['pub_id'])]
    
    early_career_quality = combined_early_quality.groupby('author')['id1'].count()
    early_career_quality_first = early_career_quality_first.groupby('author')['id1'].count()

    early_career_quality = early_career_quality.rename(f'early_career_qual_{EARLY_CAREER}').reset_index()
    early_career_quality_first = early_career_quality_first.rename(f'early_career_qual_first_{EARLY_CAREER}').reset_index()
    
    credible_authors = credible_authors.merge(early_career_quality, on='author', how='left')
    credible_authors = credible_authors.merge(early_career_quality_first, on='author', how='left')
    
    credible_authors[f'early_career_qual_{EARLY_CAREER}'] = credible_authors[f'early_career_qual_{EARLY_CAREER}'].fillna(0)
    credible_authors[f'early_career_qual_first_{EARLY_CAREER}'] = credible_authors[
        f'early_career_qual_first_{EARLY_CAREER}'].fillna(0)

# ### Early recognition

for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    for RECOGNITION_CUT in RECOGNITION_CUT_OFF_LIST:
        if RECOGNITION_CUT != EARLY_CAREER: continue
        early_career_recognition = combined[(combined.year_pub < combined.start_year + EARLY_CAREER) &
                 (combined.year_cit < combined.start_year + RECOGNITION_CUT)]
        early_career_recognition = early_career_recognition.groupby('author')['id1'].count()
        col_name = f'early_career_recognition_EC{EARLY_CAREER}_RC{RECOGNITION_CUT}'
        early_career_recognition = early_career_recognition.rename(col_name)
        early_career_recognition = early_career_recognition.reset_index()
        credible_authors = credible_authors.merge(early_career_recognition, on='author', how='left')
        credible_authors[col_name] = credible_authors[col_name].fillna(0)

credible_authors.columns

# ### Final success

combined_succ_after_15y = combined[combined.year_cit < combined.start_year + SUCCESS_CUTOFF]

# +
succ_after_15y = combined_succ_after_15y.groupby('author')['id1'].count()

succ_after_15y = succ_after_15y.rename('succ_after_15y')
succ_after_15y = succ_after_15y.reset_index()
credible_authors = credible_authors.merge(succ_after_15y, on='author', how='left')
credible_authors['succ_after_15y'] = credible_authors['succ_after_15y'].fillna(0)


# -

# ### H index

def h_index(citations):
    if len(citations) == 0: return 0
    if len(citations) == 1: return 1
    citations = sorted(citations, reverse=True)
    h_ind = 0
    for i, elem in enumerate(citations):
        if i+1 > elem:
            return i
        h_ind = i+1
    return h_ind


for param in [*EARLY_CAREER_LEN_LIST, SUCCESS_CUTOFF]:

    combined_h_index = combined[combined.year_cit < combined.start_year + param]

    combined_h_index = combined_h_index.groupby(['author', 'pub_id'])['id1'].count()

    combined_h_index = combined_h_index.reset_index()

    combined_h_index = combined_h_index.groupby('author')['id1'].apply(lambda x: h_index(x.values))

    combined_h_index = combined_h_index.rename(f'h-index_{param}')

    credible_authors = credible_authors.merge(combined_h_index.reset_index(), on='author', how='left')
    credible_authors[f'h-index_{param}'] = credible_authors[f'h-index_{param}'].fillna(0)

# +
# TODO: test h-index
# -

# # %%time
for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    early_career_publications_reduced = early_career_publications[early_career_publications.year <= 
                                                       early_career_publications.start_year + EARLY_CAREER]
    early_career_publications_ = early_career_publications_reduced.groupby('author').agg({'pub_id': 'nunique'}).reset_index()
    early_career_publications_ = early_career_publications_.rename({'pub_id':f'early_career_prod_{EARLY_CAREER}'}, axis='columns')
    credible_authors = credible_authors.merge(early_career_publications_, on='author', how='left')


# ### Early Coauthor max h-index

# +
# for each paper in EC, calculate the h-index of all its authors
# This requires extra work
# We want to calculate the h index of coauthors at the time of publishing the paper
# for this we need an extra lookup table, where we store 
# all papers - authors - h-index at the time
# 

# final_citation_count_from_ids - we merge pub data with cit data, but "inner"
# this means we will not find papers with 0 citations in this df
# these papers dont impact the h-index, so this is okay
# -

def author_h_index_in_year_X(authors, year_x):
#     print(year_x)
    combined_h = combined[(combined.year_cit < year_x) & (combined.author.isin(authors))]
    combined_h = combined_h.groupby(['author', 'pub_id']).agg({'id1': 'count'}).reset_index()
    author_hind_at_year = combined_h.groupby('author').agg({'id1': h_index}).reset_index()
    author_hind_at_year['year_pub'] = year_x
    author_hind_at_year = author_hind_at_year.rename({'id1': 'h-index'}, axis='columns')
    return author_hind_at_year


def author_h_index(author, year_x):
    combined_h = combined[(combined.year_cit < year_x) & (combined.author == author)]
    citations_count_list = combined_h.groupby(['pub_id']).agg({'id1': 'count'})['id1'].values
    return h_index(citations_count_list)


# # %%time
papers_authors = combined[['author', 'year_pub']].drop_duplicates(subset=['author', 'year_pub'])

# # %%time
all_authors_hind = pd.DataFrame(columns=['author', 'h-index', 'year_pub'])
all_authors_hind['year_pub'] = all_authors_hind['year_pub'].astype('int64')
for year_x in papers_authors.year_pub.unique():
    authors = papers_authors[papers_authors.year_pub == year_x].author.values
    author_hind_at_year = author_h_index_in_year_X(authors, year_x)
    all_authors_hind = all_authors_hind.append(author_hind_at_year)

papers_authors = papers_authors.merge(all_authors_hind, how='left')

papers_authors['h-index'] = papers_authors['h-index'].fillna(0)

for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    combined_early_coauthor = combined[(combined.year_pub < combined.start_year + EARLY_CAREER)]

    combined_early_coauthor = combined_early_coauthor.drop_duplicates(subset=['author', 'pub_id'])

    combined_early_coauthor = combined_early_coauthor[['author', 'pub_id']]

    # merging with combined not to remove coauthors that are not in their early career
    combined_early_coauthor = combined_early_coauthor.merge(combined, on='pub_id')

    combined_early_coauthor = combined_early_coauthor[combined_early_coauthor.author_x != combined_early_coauthor.author_y]
    combined_early_coauthor = combined_early_coauthor.drop_duplicates(subset=['author_x', 'author_y'])

    # papers_authors contains h-index of authors in different publishing years
    combined_early_coauthor = combined_early_coauthor.merge(papers_authors, left_on=['author_y', 'year_pub'],
                                                            right_on=['author', 'year_pub'])

    combined_early_coauthor = combined_early_coauthor.groupby('author_x')['h-index'].max().reset_index()

    combined_early_coauthor.rename({"author_x":"author", "h-index": f"early_career_coauthor_max_hindex_{EARLY_CAREER}"}, 
                                 axis='columns', inplace=True)

    combined_early_coauthor = combined_early_coauthor[['author', f"early_career_coauthor_max_hindex_{EARLY_CAREER}"]]

    credible_authors = credible_authors.merge(combined_early_coauthor, on='author', how='left')
    credible_authors[f"early_career_coauthor_max_hindex_{EARLY_CAREER}"] = credible_authors[f"early_career_coauthor_max_hindex_{EARLY_CAREER}"].fillna(0)

credible_authors.columns


# ### Early Coauthor max citations

# +
# for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
#     combined_early_coauthor = combined[(combined.year_pub < combined.start_year + EARLY_CAREER)]

#     combined_early_coauthor = combined_early_coauthor.drop_duplicates(subset=['author', 'pub_id'])

#     combined_early_coauthor = combined_early_coauthor[['author', 'pub_id']]

#     combined_early_coauthor = combined_early_coauthor.merge(combined, on='pub_id')

#     combined_early_coauthor = combined_early_coauthor[combined_early_coauthor.author_x != combined_early_coauthor.author_y]
#     combined_early_coauthor = combined_early_coauthor.drop_duplicates(subset=['author_x', 'author_y'])

#     combined_early_coauthor = combined_early_coauthor.merge(credible_authors[['author', 'succ_after_15y']], left_on='author_y', right_on='author')
#     combined_early_coauthor = combined_early_coauthor.groupby('author_x')['succ_after_15y'].max().reset_index()

#     combined_early_coauthor.rename({"author_x":"author", "succ_after_15y": f"early_career_coauthor_max_cit_{EARLY_CAREER}"}, 
#                                  axis='columns', inplace=True)

#     combined_early_coauthor = combined_early_coauthor[['author', f"early_career_coauthor_max_cit_{EARLY_CAREER}"]]

#     credible_authors = credible_authors.merge(combined_early_coauthor, on='author', how='left')
#     credible_authors[f"early_career_coauthor_max_cit_{EARLY_CAREER}"] = credible_authors[f"early_career_coauthor_max_cit_{EARLY_CAREER}"].fillna(0)
# -

# drop
def drop_list_cols(drop_list):
    credible_authors.drop(drop_list, axis=1, inplace=True)
def drop_col(df, cols):
    df.drop(cols, axis='columns', inplace=True)


# ## Save

# credible_authors.to_csv('derived-data/authors-scientific-atleast-'+str(CAREER_LENGTH)+'-year-extended.csv',
#                     index=False, encoding='utf-8')
credible_authors[credible_authors.start_year >= START_YEAR].to_csv('derived-data/authors-scientific-extended.csv',
                    index=False, encoding='utf-8')

# +
#backup
# credible_authors[credible_authors.start_year >= START_YEAR].to_csv('derived-data/authors-scientific-extended_all.csv',
#                     index=False, encoding='utf-8')
# -

credible_authors.columns


