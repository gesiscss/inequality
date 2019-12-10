import pandas as pd


class DataStore:
    '''
    DataStore class contains the required datasets in one place. 
    :credible_authors - One author per row, columns contain values grouped over the authors career life
    :cited_papers_network - One cited paper per row. Data about author, year, and which paper cited
    
    Upon first creation the DataStore class will save the basic versions of credible_authors, cited_papers_network and 
    early_career_publications. After this, call the aux methods (like team_size) to create extra columns in credible_authors 
    that depend on the early_career_len_list variable. 
    
    Creating a new instance of the DataStore class will reset the state of credible_authors_instance back 
    to the basic credible_authors.
    
    Calling to_dataframe() gives a copy of the credible_authors_instance. This way we can have many copies for different 
    values of early_career_len_list.
    
    '''
    credible_authors = None
    cited_papers_network = None
    early_career_publications = None
    all_papers_num_cit = None

    authorPublicationData = None
    authorCitationsData = None

    def __init__(self):
        self.credible_authors_instance = None
        self._refresh_data_cache()

    def _refresh_data_cache(self):
        if DataStore.credible_authors is None:
            print("load publication data from CSV")
            DataStore.authorPublicationData = pd.read_csv('./data/author_publications_2017_asiansAsNone.txt')
            DataStore.authorPublicationData.drop_duplicates(subset=['author', 'pub_id'], inplace=True)
            # TODO: Remove this line! :)
            # DataStore.authorPublicationData = DataStore.authorPublicationData.head(10000)
            print("load citation data from CSV")
            DataStore.authorCitationsData = pd.read_csv('./data/citations_2017_asiansAsNone.txt')
            DataStore.authorCitationsData.drop_duplicates(inplace=True)

            DataStore.credible_authors = DataStore._create_credible_authors()
            DataStore._add_gender()

        self.credible_authors_instance = DataStore.credible_authors.copy()

        if DataStore.cited_papers_network is None:
            print("cited_papers_network - create")
            DataStore._create_cited_papers_network()
            DataStore._clean_cited_papers_network()
            print("cited_papers_network - add start year")
            DataStore._add_start_year()

        if DataStore.early_career_publications is None:
            print("early_career_publications - create")
            DataStore._create_early_career_publications()

        if DataStore.all_papers_num_cit is None:
            print("all_papers_num_cit - create")
            DataStore._create_all_papers_num_cit()

    @classmethod
    def _create_credible_authors(cls):
        group_by_author = cls.authorPublicationData.groupby(['author'])

        group_by_author_min_year_data = group_by_author['year'].min()
        group_by_author_max_year_data = group_by_author['year'].max()
        group_by_author_count_publications_data = group_by_author['pub_id'].count()
        author_grouped_data = group_by_author_min_year_data.to_frame(name='start_year')
        author_grouped_data['end_year'] = group_by_author_max_year_data
        author_grouped_data['total_num_pub'] = group_by_author_count_publications_data
        author_grouped_data = author_grouped_data.reset_index()
        author_grouped_data = author_grouped_data.drop_duplicates()
        author_grouped_data = author_grouped_data.dropna(how='any')

        author_grouped_data["career_length"] = author_grouped_data['end_year'] - author_grouped_data['start_year'] + 1

        return author_grouped_data

    @classmethod
    def _add_gender(cls):
        if 'gender' not in cls.credible_authors.columns:
            df = cls.credible_authors
            gender = pd.read_csv('./data/name_gender_2017_asiansAsNone_nodup.txt')

            df = df.set_index('author')
            df['gender'] = gender.set_index('name')['gender']
            cls.credible_authors = df.reset_index()

    @classmethod
    def _create_cited_papers_network(cls):
        # TODO: Optimize this with the all_papers_num_cit, its almost the same
        final_citation_count_from_ids = cls.authorPublicationData.merge(cls.authorCitationsData,
                                                                        left_on='pub_id',
                                                                        right_on='id2', how='inner',
                                                                        suffixes=('_pub', '_cit'))
        cls.cited_papers_network =  final_citation_count_from_ids

    @classmethod
    def _clean_cited_papers_network(cls):
        final_citation_count_from_ids = cls.cited_papers_network
        final_citation_count_from_ids.drop_duplicates(inplace=True)

        cit_wrong = final_citation_count_from_ids[
            final_citation_count_from_ids.year_pub > final_citation_count_from_ids.year_cit].index

        final_citation_count_from_ids.drop(cit_wrong, inplace=True)
        final_citation_count_from_ids = final_citation_count_from_ids.drop_duplicates(
            subset=['author', 'id1', 'id2', 'year_cit'])
        cls.cited_papers_network = final_citation_count_from_ids

    @classmethod
    def _add_start_year(cls):
        if 'start_year' not in cls.cited_papers_network.columns:
            df = cls.cited_papers_network
            df = df.merge(cls.credible_authors[['author', 'start_year']], on='author',
                          how='inner')
            df.drop_duplicates(subset=['author', 'id1', 'id2'], inplace=True)
            cls.cited_papers_network = df

    def team_size(self, early_career_len_list):
        print(f"team_size for ec in {early_career_len_list}")
        ec_ = early_career_len_list[0]
        if f'team_size_median_{ec_}' not in self.credible_authors_instance.columns:
            for EARLY_CAREER in early_career_len_list:
                early_career_publications_filtered = DataStore.early_career_publications[(
                        DataStore.early_career_publications.year < DataStore.early_career_publications.start_year + EARLY_CAREER)]
                paper_team_size = early_career_publications_filtered.groupby('pub_id').agg(
                    {'author': 'nunique'}).reset_index()
                paper_team_size = paper_team_size.rename({'author': f'team_size_{EARLY_CAREER}'}, axis='columns')
                early_career_publications_filtered = early_career_publications_filtered.merge(paper_team_size,
                                                                                              on='pub_id', how='left')
                team_size_median = early_career_publications_filtered.groupby('author').agg(
                    {f'team_size_{EARLY_CAREER}': 'median'}).reset_index()
                team_size_median = team_size_median.rename(
                    {f'team_size_{EARLY_CAREER}': f'team_size_median_{EARLY_CAREER}'}, axis='columns')
                team_size_mean = early_career_publications_filtered.groupby('author').agg(
                    {f'team_size_{EARLY_CAREER}': 'mean'}).reset_index()
                team_size_mean = team_size_mean.rename({f'team_size_{EARLY_CAREER}': f'team_size_mean_{EARLY_CAREER}'},
                                                       axis='columns')
                self.credible_authors_instance = self.credible_authors_instance.merge(team_size_median, on='author',
                                                                                      how='left')
                self.credible_authors_instance = self.credible_authors_instance.merge(team_size_mean, on='author',
                                                                                      how='left')
        return self

    def run_all_credible_author_methods(self, early_career_len_list):
        self.team_size(early_career_len_list)

        return self

    @classmethod
    def _create_early_career_publications(cls):
        early_career_publications = cls.authorPublicationData.merge(cls.credible_authors[['author', 'start_year']],
                                                                    on='author',
                                                                    how='left')
        cls.early_career_publications = early_career_publications

    def to_dataframe(self):
        return self.credible_authors_instance.copy()

    @staticmethod
    # TODO: fix this name
    def to_csv(filename='author-publications.csv'):
        DataStore.credible_authors.to_csv(f'derived-data/{filename}', index=False)

    @classmethod
    def _create_all_papers_num_cit(cls):
        cls.uncited_papers_network = cls.authorPublicationData.merge(cls.authorCitationsData, left_on='pub_id',
                                                                     right_on='id2', how='left',
                                                                     suffixes=('_pub', '_cit'))
        cls.uncited_papers_network = cls.uncited_papers_network.drop_duplicates(subset=['pub_id', 'id1'])
        cls.all_papers_num_cit = cls.uncited_papers_network.groupby('pub_id').agg({'id1': 'count'})
