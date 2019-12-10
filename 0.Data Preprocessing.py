# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"pycharm": {"is_executing": false}}
import pandas as pd
import numpy as np

import calculate
from calculate import gini, h_index

# +
# Specify how long is the early career. Impacts which papers we take into account for early productivity and quality
EARLY_CAREER_LEN = 3
EARLY_CAREER_LEN_LIST = [3, 5, 7, 9, 11, 12]
EARLY_CAREER_LEN_LIST = [3]
# For early career work, when do we stop counting citations. Impacts recognition
RECOGNITION_CUT_OFF_LIST = [3, 5, 7, 9, 11, 12]
RECOGNITION_CUT_OFF_LIST = [3]
# Success after 15 years. Impacts when we stop counting citations
SUCCESS_CUTOFF = 15
SUCCESS_CUTOFF_LIST = [10, 15]
# Length of observed career for dropouts
# (1-3), middle career (4-9), late career (10-15)

# TODO: for multiple dropout intervals code does not work!!!
# CAREER_LENGTH_DROPOUTS_LIST = [ (0, 15), (0, 3), (4, 9), (10, 15)] #,
CAREER_LENGTH_DROPOUTS_LIST = [(0, 15)]
# CAREER_LENGTH_DROPOUTS = 15
INACTIVE_TIME_DROPOUTS = 10
INACTIVE_TIME_DROPOUTS_LIST = 5

# Specify the first and last year we consider in our analysis
START_YEAR = 1970
LAST_START_YEAR = 2000


# -

def get_start_years(START_YEAR, LAST_START_YEAR):
    all_years = credible_authors.start_year.unique()
    start_years = [year for year in all_years if START_YEAR <= year <= LAST_START_YEAR]
    start_years = sorted(start_years)
    return start_years


# ## 1. Load data

# +
# cheat load
# credible_authors = pd.read_csv('derived-data/authors-scientific-extended.csv')
# -

# read csv raw data
# publications
authorPublicationData = pd.read_csv('./data/author_publications_2017_asiansAsNone.txt')
# citations
authorCitationsData = pd.read_csv('./data/citations_2017_asiansAsNone.txt')
# arxiv
arxiv_pubid = pd.read_csv('derived-data/arxiv_pubid_2017.csv', header=None, names=['pub_id'])
# venue data
publication_venues_rank = pd.read_csv('derived-data/publication-venues-rank.csv')

# +
# remove duplicates and remove arxiv
print(authorPublicationData.shape)
authorPublicationData.drop_duplicates(subset=['author', 'pub_id'], inplace=True)
print(authorPublicationData.shape)
authorPublicationData = authorPublicationData.loc[~authorPublicationData.pub_id.isin(arxiv_pubid['pub_id'])]
print(authorPublicationData.shape)

print(authorCitationsData.shape)
authorCitationsData.drop_duplicates(inplace=True)
print(authorCitationsData.shape)
authorCitationsData = authorCitationsData.loc[(~authorCitationsData.id1.isin(arxiv_pubid['pub_id']))]
authorCitationsData = authorCitationsData.loc[(~authorCitationsData.id2.isin(arxiv_pubid['pub_id']))]
print(authorCitationsData.shape)
# -

print('Authors#      - ', authorPublicationData['author'].nunique())
print('Years#        - ', authorPublicationData['year'].nunique())
print('Publications# - ', authorPublicationData['pub_id'].nunique())

# ## 2. Career length and gender

# +
groupByAuthor = authorPublicationData.groupby(['author'])

groupByAuthorMinYearData = groupByAuthor['year'].min()
groupByAuthorMaxYearData = groupByAuthor['year'].max()
groupByAuthorCountPublicationsData = groupByAuthor['pub_id'].count()

authorGroupedData = groupByAuthorMinYearData.to_frame(name='start_year')
authorGroupedData['end_year'] = groupByAuthorMaxYearData
authorGroupedData['total_num_pub'] = groupByAuthorCountPublicationsData
authorGroupedData = authorGroupedData.reset_index()
print('Total rows -                ', authorGroupedData.shape)

authorGroupedData = authorGroupedData.drop_duplicates()
print('After removing duplicates - ', authorGroupedData.shape)

authorGroupedData = authorGroupedData.dropna(how='any')
print("After droping na -          ", authorGroupedData.shape)

authorGroupedData.head()
# -

# Adding 1 here to have career length be at least one. So 3 years career means year1, year2, year3.
authorGroupedData["career_length"] = authorGroupedData['end_year'] - authorGroupedData['start_year'] + 1

credible_authors = authorGroupedData

# ### Gender

# +
gender = pd.read_csv('./data/name_gender_2017_asiansAsNone_nodup.txt')
credible_authors = credible_authors.merge(gender, left_on='author', right_on='name', how='left')
credible_authors.drop('name', axis=1, inplace=True)

print(credible_authors.gender.value_counts())
gender.head()


# -

# ### Save filtered data about authors, and cleaned publications

def filter_cred_authors(START_YEAR, LAST_START_YEAR):
    return credible_authors[
        (credible_authors.start_year >= START_YEAR) & (credible_authors.start_year <= LAST_START_YEAR)]


filter_cred_authors(START_YEAR, LAST_START_YEAR).to_csv('derived-data/authors-scientific.csv', index=False,
                                                        encoding='utf-8')
credible_authors.head()

authorPublicationData.to_csv('derived-data/author-publications.csv', index=False)

authorPublicationData.shape

# ## 3. Prepare DFs 

# ### DF1 - Publications and Citations, no uncited papers

# For every author and paper => every citation to the paper
# Doesnt contain uncited papers
# Contains multiple authors per paper
# This is good for per author analysis
# This is bad for per paper analysis
publications_citations_no_uncited = authorPublicationData.merge(authorCitationsData, left_on='pub_id',
                                                                right_on='id2', how='inner', suffixes=('_pub', '_cit'))
publications_citations_no_uncited = publications_citations_no_uncited.merge(credible_authors[['author', 'start_year']],
                                                                            on='author', how='inner')

# remove cited before published
publications_citations_no_uncited = publications_citations_no_uncited[
    publications_citations_no_uncited.year_pub <= publications_citations_no_uncited.year_cit]

# remove duplicate authors
# no uncited papers
paper_paper_citations = publications_citations_no_uncited[['id1', 'id2', 'year_pub', 'year_cit']]
paper_paper_citations = paper_paper_citations.drop_duplicates(subset=['id1', 'id2'])
paper_total_citations = paper_paper_citations.groupby('id2')['id1'].count()

# ### DF2 - Publications with uncited papers

# Contains uncited papers
# Good for early career publication related analysis
publications_start_year = authorPublicationData.merge(credible_authors[['author', 'start_year']], on='author',
                                                      how='inner')

# ### DF3 - Author order in publications

# + {"pycharm": {"name": "#%%\n"}}
publications_first_author = pd.read_csv('derived-data/publication_authors_order_2017.csv')

# + {"pycharm": {"name": "#%%\n"}}
publications_first_author = publications_first_author.merge(credible_authors[['author', 'start_year']],
                                                            left_on='first_author',
                                                            right_on='author', how='left')
publications_first_author = publications_first_author.drop('first_author', axis='columns')
# -

# ### Author yearly citations and publications

# number of citations an author receives per year
author_yearly_citations = publications_citations_no_uncited.groupby(['author', 'year_cit'])['id1'].count()
author_yearly_citations = author_yearly_citations.reset_index()
author_yearly_citations = author_yearly_citations.rename(columns={'id1': 'num_cit', 'year_cit': 'year'})
author_yearly_citations[['author', 'year', 'num_cit']].to_csv('derived-data/authors-perYear-citations.csv', index=False)

author_yearly_publications = authorPublicationData.groupby(['author', 'year'])['pub_id'].count().reset_index()
author_yearly_publications = author_yearly_publications.rename(columns={'pub_id': 'num_pub'})

# ## Publication and citation based analysis - DF1

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ### 1: Early quality

# + {"pycharm": {"name": "#%%\n"}, "hidden": true}
for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    combined_early_quality = publications_citations_no_uncited[
        (publications_citations_no_uncited.year_pub < publications_citations_no_uncited.start_year + EARLY_CAREER) &
        (publications_citations_no_uncited.year_cit < publications_citations_no_uncited.start_year + SUCCESS_CUTOFF)]

    author_order_early = publications_first_author[
        (publications_first_author.year < publications_first_author.start_year + EARLY_CAREER)]
    early_career_quality_first = combined_early_quality.loc[
        combined_early_quality['pub_id'].isin(author_order_early['pub_id'])]

    early_career_quality = combined_early_quality.groupby('author')['id1'].count()
    early_career_quality_first = early_career_quality_first.groupby('author')['id1'].count()

    early_career_quality = early_career_quality.rename(f'early_career_qual_{EARLY_CAREER}').reset_index()
    early_career_quality_first = early_career_quality_first.rename(
        f'early_career_qual_first_{EARLY_CAREER}').reset_index()

    credible_authors = credible_authors.merge(early_career_quality, on='author', how='left')
    credible_authors = credible_authors.merge(early_career_quality_first, on='author', how='left')

    credible_authors[f'early_career_qual_{EARLY_CAREER}'] = credible_authors[
        f'early_career_qual_{EARLY_CAREER}'].fillna(0)
    credible_authors[f'early_career_qual_first_{EARLY_CAREER}'] = credible_authors[
        f'early_career_qual_first_{EARLY_CAREER}'].fillna(0)

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ### 2: Early recognition

# + {"pycharm": {"name": "#%%\n"}, "hidden": true}
for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    for RECOGNITION_CUT in RECOGNITION_CUT_OFF_LIST:
        if RECOGNITION_CUT != EARLY_CAREER: continue
        early_career_recognition = publications_citations_no_uncited[
            (publications_citations_no_uncited.year_pub < publications_citations_no_uncited.start_year + EARLY_CAREER) &
            (
                        publications_citations_no_uncited.year_cit < publications_citations_no_uncited.start_year + RECOGNITION_CUT)]
        early_career_recognition = early_career_recognition.groupby('author')['id1'].count()
        col_name = f'early_career_recognition_EC{EARLY_CAREER}_RC{RECOGNITION_CUT}'
        early_career_recognition = early_career_recognition.rename(col_name)
        early_career_recognition = early_career_recognition.reset_index()
        credible_authors = credible_authors.merge(early_career_recognition, on='author', how='left')
        credible_authors[col_name] = credible_authors[col_name].fillna(0)


# -

# ## Publication based analysis - DF2

# ### 1: Label authors that drop out

def list_append(lst, item):
    lst.append(item)
    return lst


def append_new_only(lst, item):
    if item not in lst:
        lst.append(item)
    return lst


# + {"pycharm": {"name": "#%%\n"}}
def get_author_avg_max_absence(CAREER_LENGTH_DROPOUTS_LIST, INACTIVE_TIME_DROPOUTS, credible_authors):
    for start, end in CAREER_LENGTH_DROPOUTS_LIST:
        pubs_grouped = publications_start_year[
            (publications_start_year.year >= publications_start_year.start_year + start) &
            (publications_start_year.year <= publications_start_year.start_year + end)]
        # for every 2 consecutive years the author has published find a difference (absence time)
        # we artificially add one value: career start + 15, as a limiter of our observation window
        # we add the limiter only if the author did not publish in this year
        pubs_grouped = pubs_grouped.groupby('author').agg({'year': lambda x: sorted(list(x))})
        pubs_grouped['year'] = pubs_grouped['year'].apply(lambda x: sorted(append_new_only(x, x[0] + 15)))
        pubs_grouped['absence_list'] = pubs_grouped['year'].apply(np.diff)
        pubs_grouped[f'max_absence_{start}_{end}'] = pubs_grouped['absence_list'].apply(max)
        pubs_grouped[f'avg_absence_{start}_{end}'] = pubs_grouped['absence_list'].apply(np.mean)
        pubs_grouped.reset_index(inplace=True)

        credible_authors = credible_authors.merge(pubs_grouped[
                                                      ['author', f'max_absence_{start}_{end}',
                                                       f'avg_absence_{start}_{end}']], on='author', how='left')
        credible_authors[f'dropped_after_{INACTIVE_TIME_DROPOUTS}'] = credible_authors[
            f'max_absence_{start}_{end}'].apply(
            lambda x: False if x < INACTIVE_TIME_DROPOUTS else True)


get_author_avg_max_absence(CAREER_LENGTH_DROPOUTS_LIST, INACTIVE_TIME_DROPOUTS, credible_authors)
# -

# ### 2: Team size

# + {"pycharm": {"name": "#%%\n"}}
for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    publications_early = publications_start_year[(
            publications_start_year.year < publications_start_year.start_year + EARLY_CAREER)]
    paper_team_size = publications_early.groupby('pub_id').agg({'author': 'nunique'}).reset_index()
    paper_team_size = paper_team_size.rename({'author': f'team_size_{EARLY_CAREER}'}, axis='columns')
    publications_early = publications_early.merge(paper_team_size, on='pub_id', how='left')
    team_size_median = publications_early.groupby('author').agg({f'team_size_{EARLY_CAREER}': 'median'}).reset_index()
    team_size_median = team_size_median.rename({f'team_size_{EARLY_CAREER}': f'team_size_median_{EARLY_CAREER}'},
                                               axis='columns')
    team_size_mean = publications_early.groupby('author').agg({f'team_size_{EARLY_CAREER}': 'mean'}).reset_index()
    team_size_mean = team_size_mean.rename({f'team_size_{EARLY_CAREER}': f'team_size_mean_{EARLY_CAREER}'},
                                           axis='columns')

    credible_authors = credible_authors.merge(team_size_median, on='author', how='left')
    credible_authors = credible_authors.merge(team_size_mean, on='author', how='left')
# -

# ### 3: Early degree

# + {"pycharm": {"name": "#%%\n"}}
for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    combined_early_degree = publications_start_year[
        (publications_start_year.year < publications_start_year.start_year + EARLY_CAREER)]

    combined_early_degree = combined_early_degree.drop_duplicates(subset=['author', 'pub_id'])

    combined_early_degree = combined_early_degree[['author', 'pub_id']]

    combined_early_degree = combined_early_degree.merge(publications_start_year, on='pub_id')

    combined_early_degree = combined_early_degree[combined_early_degree.author_x != combined_early_degree.author_y]
    combined_early_degree = combined_early_degree.drop_duplicates(subset=['author_x', 'author_y'])

    combined_early_degree = combined_early_degree.groupby('author_x')['author_y'].count().reset_index()

    combined_early_degree.rename({"author_x": "author", "author_y": f"early_career_degree_{EARLY_CAREER}"},
                                 axis='columns', inplace=True)

    credible_authors = credible_authors.merge(combined_early_degree, on='author', how='left')
    credible_authors[f"early_career_degree_{EARLY_CAREER}"] = credible_authors[
        f"early_career_degree_{EARLY_CAREER}"].fillna(0)

# ### Venues

early_career_venues = publications_start_year.merge(publication_venues_rank[[
    'pub_id', 'h5_index', 'ranking', 'deciles', 'quantiles']], on='pub_id', how='inner')


# +
# TODO including the MAX and MIN values as missing. Check this. also what to do with ranking?
def quantile_binary(quant):
    return quant <= 3


for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    #     EARLY_CAREER = 3
    early_career_venues_ec = early_career_venues[
        early_career_venues.year < early_career_venues.start_year + EARLY_CAREER]
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
    credible_authors[f'quantiles_bin_{EARLY_CAREER}'] = credible_authors[f'quantiles_min_{EARLY_CAREER}'].apply(
        quantile_binary)

    # credible_authors[f'ranking_{EARLY_CAREER}'] = credible_authors[f'ranking_{EARLY_CAREER}'].fillna(0)

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ### Number of early publications - first author (DF3)

# + {"pycharm": {"name": "#%%\n"}}
# number of first author publications
for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
    publications_first_author_early = publications_first_author[
        (publications_first_author.year < publications_first_author.start_year + EARLY_CAREER)]
    publications_first_author_early = publications_first_author_early.groupby('author').agg(
        {'pub_id': 'count'}).reset_index()
    publications_first_author_early.rename({'pub_id': f'ec_first_auth_{EARLY_CAREER}'}, axis='columns', inplace=True)
    credible_authors = credible_authors.merge(publications_first_author_early, on='author', how='left')
    credible_authors[f'ec_first_auth_{EARLY_CAREER}'] = credible_authors[f'ec_first_auth_{EARLY_CAREER}'].fillna(0)
# -

# ## Counts Dataframe 
# For every author, construct a 15 years time frame and count metrics for every career age

start_years = get_start_years(START_YEAR, LAST_START_YEAR)

# +
# create publication and citation dfs for first author
author_year_numPub_first = publications_first_author.groupby(['author', 'year'])['pub_id'].count().reset_index()
author_year_numPub_first = author_year_numPub_first.rename(columns={'pub_id': 'num_pub'})

publications_citations_no_uncited_first = publications_citations_no_uncited.merge(
    publications_first_author[['author', 'pub_id']], how='inner')
citations_year_auth_first = publications_citations_no_uncited_first.groupby(['author', 'year_cit'])['id1'].count()

citations_year_auth_first = citations_year_auth_first.reset_index()
citations_year_auth_first = citations_year_auth_first.rename(columns={'id1': 'num_cit', 'year_cit': 'year'})
# -

print(publications_citations_no_uncited_first.shape)
print(publications_citations_no_uncited.shape)


# + {"code_folding": [1, 21, 35, 66, 70]}
# create a dataframe with 15 career age entries for every author
def create_counts_df(credible_authors, citations_year_auth_df, author_year_numPub_df):
    counts0 = credible_authors[['author', 'start_year']].copy()
    # filter out start years
    counts0 = counts0[counts0['start_year'].isin(start_years)]
    counts0['year'] = counts0['start_year'].apply(lambda x: [x + i for i in range(0, 15)])
    counts = pd.DataFrame(counts0['year'].tolist(), index=counts0['author']).stack().reset_index(
        level=1, drop=True).reset_index(name='year')[['year', 'author']]
    counts = counts.merge(credible_authors[['author', 'start_year', 'end_year', 'gender']], on='author', how='inner')
    counts['career_age'] = counts['year'] - counts['start_year'] + 1
    counts['year'] = counts['year'].astype('int32')

    counts['career_duration'] = counts['end_year'] - counts['start_year'] + 1
    
    counts = add_absolute_counts(counts, citations_year_auth_df, author_year_numPub_df)
    
    counts = add_cumulative_counts(counts, 'num_cit')
    counts = add_cumulative_counts(counts, 'num_pub')
    
    return counts

def add_absolute_counts(counts_df, citations_year_auth_df, author_year_numPub_df):
    # merge in publications
    counts_df = counts_df.merge(author_year_numPub_df, on=['author', 'year'], how='left')
    counts_df['num_pub'] = counts_df['num_pub'].fillna(0)
    # merge in citations
    counts_df = counts_df.merge(citations_year_auth_df, on=['author', 'year'], how='left')
    counts_df['num_cit'] = counts_df['num_cit'].fillna(0)
    return counts_df

def add_cumulative_counts(counts_df, feature):
    counts_df = calculate.calculate_cumulative_for_authors(counts_df, feature)
    return counts_df

# add citation and publication window features to citations df
def add_citation_window_counts(counts_df, combined_df, WINDOW_SIZE):
    """
    Adds 2 columns to the counts dataframe: win_num_cit, win_num_pub
    win_num_pub: count publications for career ages, with forward looking windows of size WINDOW_SIZE
    
    win_num_cit: for a publication window defined by win_num_pub, count citations in a sliding window
    that starts at publication year and extends to pub_year+WINDOW_SIZE, non inclusive
    """
    shift = -(WINDOW_SIZE - 1)
    # citations window
    df_list = []
    for year in start_years:
        df_year = combined_df[combined_df.start_year == year]
        for y in range(year, year + 13):  # y is the first year we count for
            df_window = df_year[(df_year.year_pub >= y) & (df_year.year_pub < y + WINDOW_SIZE) &
                                (df_year.year_cit >= y) & 
                                (df_year.year_cit < df_year.year_pub + WINDOW_SIZE)]
            df_window = df_window.groupby('author').agg({'id1': 'count'}).reset_index()
            df_window['year'] = y
            df_window = df_window.rename({'id1': 'win_num_cit'}, axis=1)
            df_list.append(df_window)
    df_cit_window = pd.concat(df_list).sort_values(by=['author', 'year'])
    counts_df = counts_df.merge(df_cit_window, on=['author', 'year'], how='left')
    counts_df['win_num_cit'] = counts_df['win_num_cit'].fillna(0)
    
    counts_df['win_num_pub'] = counts_df.groupby('author')['num_pub'].transform(
        lambda x: x.rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE).sum().shift(shift))
    
    return counts_df


def save_counts(counts_df, WINDOW_SIZE, ext=''):
    counts_df.to_csv(f'derived-data/citations_window_{WINDOW_SIZE}{ext}.csv', index=None)


def make_counts_file(base_df, publications_citations_no_uncited, WINDOW_SIZE, file_ext=''):
    counts_df = base_df.copy(deep=True)
    counts_df = add_citation_window_counts(base_df, publications_citations_no_uncited, WINDOW_SIZE)

    save_counts(counts_df, WINDOW_SIZE, file_ext)
    return counts_df


# -

# %%time
base_df = create_counts_df(credible_authors, author_yearly_citations, author_yearly_publications)
base_df_first = create_counts_df(credible_authors, citations_year_auth_first, author_year_numPub_first)

# %%time
# all authors
WINDOW_SIZE = 3
counts_df = make_counts_file(base_df, publications_citations_no_uncited, WINDOW_SIZE)
# first author
counts_df_first = make_counts_file(base_df_first, publications_citations_no_uncited_first, WINDOW_SIZE, file_ext='_first')

# %%time
WINDOW_SIZE = 5
# all authors
counts_df_5 = make_counts_file(base_df, publications_citations_no_uncited, WINDOW_SIZE)
# first author
counts_df_first_5 = make_counts_file(base_df_first, publications_citations_no_uncited_first, WINDOW_SIZE, file_ext='_first')
#TODO write tests for counts

# ### Early, mid and late papers analysis - citations

# + {"code_folding": []}
# TODO TEST THIS?!
def add_fine_grained_citation_counts(counts):
    # publish_years = [[0,3], [3,6], [6,9], [0,1], [3,4], [6,7]]
    publish_years = [[i, i + 1] for i in range(0, 15)]
    for start, end in publish_years:
        first_3 = publications_citations_no_uncited[
            (publications_citations_no_uncited.year_pub >= publications_citations_no_uncited.start_year + start) &
            (publications_citations_no_uncited.year_pub < publications_citations_no_uncited.start_year + end)]
        first_3 = first_3.groupby(['author', 'year_cit']).agg({'id1': 'count'}).reset_index()
        first_3 = first_3.rename({'year_cit': 'year', 'id1': f'ec_cit_{start}_{end}'}, axis=1)
        counts = counts.merge(first_3, on=['author', 'year'], how='left')
        counts[f'ec_cit_{start}_{end}'] = counts[f'ec_cit_{start}_{end}'].fillna(0)
    for start, end in publish_years:
        counts[f'ec_cit_{start}_{end}_cum'] = counts.sort_values(['author', 'career_age']).groupby('author')[
            f'ec_cit_{start}_{end}'].transform(pd.Series.cumsum)


# -

# ### Final success

succ_after_15y = publications_citations_no_uncited[
    publications_citations_no_uncited.year_cit < publications_citations_no_uncited.start_year + SUCCESS_CUTOFF]

# +
succ_after_15y = succ_after_15y.groupby('author')['id1'].count()

succ_after_15y = succ_after_15y.rename('succ_after_15y')
succ_after_15y = succ_after_15y.reset_index()
credible_authors = credible_authors.merge(succ_after_15y, on='author', how='left')
credible_authors['succ_after_15y'] = credible_authors['succ_after_15y'].fillna(0)
# -


# ### H index




for param in [*EARLY_CAREER_LEN_LIST, SUCCESS_CUTOFF]:
    combined_h_index = publications_citations_no_uncited[
        publications_citations_no_uncited.year_cit < publications_citations_no_uncited.start_year + param]

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
    early_career_publications_reduced = publications_start_year[publications_start_year.year <=
                                                                publications_start_year.start_year + EARLY_CAREER]
    early_career_publications_ = early_career_publications_reduced.groupby('author').agg(
        {'pub_id': 'nunique'}).reset_index()
    early_career_publications_ = early_career_publications_.rename({'pub_id': f'early_career_prod_{EARLY_CAREER}'},
                                                                   axis='columns')
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
    combined_h = publications_citations_no_uncited[
        (publications_citations_no_uncited.year_cit < year_x) &
        (publications_citations_no_uncited.author.isin(authors))]
    combined_h = combined_h.groupby(['author', 'pub_id']).agg({'id1': 'count'}).reset_index()
    author_hind_at_year = combined_h.groupby('author').agg({'id1': h_index}).reset_index()
    author_hind_at_year['year_pub'] = year_x
    author_hind_at_year = author_hind_at_year.rename({'id1': 'h-index'}, axis='columns')
    return author_hind_at_year


def author_h_index(author, year_x):
    combined_h = publications_citations_no_uncited[
        (publications_citations_no_uncited.year_cit < year_x) &
        (publications_citations_no_uncited.author == author)]
    citations_count_list = combined_h.groupby(['pub_id']).agg({'id1': 'count'})['id1'].values
    return h_index(citations_count_list)


# # %%time
papers_authors = publications_citations_no_uncited[['author', 'year_pub']].drop_duplicates(
    subset=['author', 'year_pub'])

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
    combined_early_coauthor = publications_start_year[
        (publications_start_year.year_pub < publications_start_year.start_year + EARLY_CAREER)]

    combined_early_coauthor = combined_early_coauthor.drop_duplicates(subset=['author', 'pub_id'])

    combined_early_coauthor = combined_early_coauthor[['author', 'pub_id']]

    # merging with combined_df not to remove coauthors that are not in their early career
    combined_early_coauthor = combined_early_coauthor.merge(publications_start_year, on='pub_id')

    combined_early_coauthor = combined_early_coauthor[
        combined_early_coauthor.author_x != combined_early_coauthor.author_y]
    combined_early_coauthor = combined_early_coauthor.drop_duplicates(subset=['author_x', 'author_y'])

    # papers_authors contains h-index of authors in different publishing years
    combined_early_coauthor = combined_early_coauthor.merge(papers_authors, left_on=['author_y', 'year_pub'],
                                                            right_on=['author', 'year_pub'])

    combined_early_coauthor = combined_early_coauthor.groupby('author_x')['h-index'].max().reset_index()

    combined_early_coauthor.rename(
        {"author_x": "author", "h-index": f"early_career_coauthor_max_hindex_{EARLY_CAREER}"},
        axis='columns', inplace=True)

    combined_early_coauthor = combined_early_coauthor[['author', f"early_career_coauthor_max_hindex_{EARLY_CAREER}"]]

    credible_authors = credible_authors.merge(combined_early_coauthor, on='author', how='left')
    credible_authors[f"early_career_coauthor_max_hindex_{EARLY_CAREER}"] = credible_authors[
        f"early_career_coauthor_max_hindex_{EARLY_CAREER}"].fillna(0)

# +
# for year in EARLY_CAREER_LEN_LIST[1:]:
#     credible_authors[f'citation_increase_{year}_{EARLY_CAREER_LEN}'] = credible_authors[
#         f'early_career_recognition_EC{year}_RC{year}'] - credible_authors[f'early_career_recognition_EC{EARLY_CAREER_LEN}_RC{EARLY_CAREER_LEN}']
#     credible_authors[f'h_index_increase_{year}_{EARLY_CAREER_LEN}'] = credible_authors[f'h-index_{year}'] - credible_authors[f'h-index_{EARLY_CAREER_LEN}']
# -

EARLY_CAREER_LEN_LIST

for year in EARLY_CAREER_LEN_LIST:
    credible_authors[f'citation_increase_15_{year}'] = credible_authors['succ_after_15y'] - credible_authors[
        f'early_career_recognition_EC{year}_RC{year}']
    credible_authors[f'h_index_increase_{year}_{EARLY_CAREER}'] = credible_authors[
                                                                      f'h-index_{year}'] - credible_authors[
                                                                      f'h-index_{EARLY_CAREER}']
    credible_authors[f'h_index_increase_15_{EARLY_CAREER}'] = credible_authors[
                                                                  f'h-index_15'] - credible_authors[
                                                                  f'h-index_{EARLY_CAREER}']


# ### Early Coauthor max citations

# +
# for EARLY_CAREER in EARLY_CAREER_LEN_LIST:
#     combined_early_coauthor = combined_df[(combined_df.year_pub < combined_df.start_year + EARLY_CAREER)]

#     combined_early_coauthor = combined_early_coauthor.drop_duplicates(subset=['author', 'pub_id'])

#     combined_early_coauthor = combined_early_coauthor[['author', 'pub_id']]

#     combined_early_coauthor = combined_early_coauthor.merge(combined_df, on='pub_id')

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

credible_authors[credible_authors.start_year >= START_YEAR].to_csv('derived-data/authors-scientific-extended.csv',
                                                                   index=False, encoding='utf-8')

credible_authors.columns
