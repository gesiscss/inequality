import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calculate
import scipy.stats as stats

def plot_gender_numcum(ax, cohort_duration, selected_cumnum_df, year):
    ax.plot(cohort_duration, selected_cumnum_df["mean_f"].values,  label='women', color="red")
    ax.fill_between(cohort_duration, selected_cumnum_df["mean_f"].values-selected_cumnum_df["sem_f"].values, 
                    selected_cumnum_df["mean_f"].values+selected_cumnum_df["sem_f"].values,
					alpha=0.2, edgecolor='red', facecolor='red',
					linewidth=4, linestyle='dashdot', antialiased=True)
    
    ax.plot(cohort_duration, selected_cumnum_df["mean_m"].values,  label='men', color="blue")
    ax.fill_between(cohort_duration, selected_cumnum_df["mean_m"].values-selected_cumnum_df["sem_m"].values, 
                    selected_cumnum_df["mean_m"].values+selected_cumnum_df["sem_m"].values,
					alpha=0.2, edgecolor='blue', facecolor='blue',
					linewidth=4, linestyle='dashdot', antialiased=True)

    ax.plot(cohort_duration, selected_cumnum_df["mean_n"].values,  label='unknown', color="grey")
    ax.fill_between(cohort_duration, selected_cumnum_df["mean_n"].values-selected_cumnum_df["sem_n"].values, 
                    selected_cumnum_df["mean_n"].values+selected_cumnum_df["sem_n"].values,
					alpha=0.2, edgecolor='grey', facecolor='grey',
					linewidth=4, linestyle='dashdot', antialiased=True) 
    return ax
	
	
def plot_cohort_analysis_on(years, authorStartEndCareerData, data, criterion, criteria_display):
    # data - groupByAuthorYearData (or) groupCitationsByAuthorYearData
    # criterion - 'cum_num_pub' (or) 'cum_num_cit'
    # criteria_display - 'Cumulative Publication Count' (or) 'Cumulative Citation Count'
    #cohort_len = []
    
    print(authorStartEndCareerData.head(n=1))
    gini_per_cohort = pd.DataFrame(index=years)
    cumnum_per_cohort = pd.DataFrame(index=years)


    #fig2, ax2 = plt.subplots()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)


    max_years = 15
    # limit plot to the N years during which we follow a cohort
    cohort_duration = np.arange(max_years)


    # 30 years
    fig5 = plt.figure(figsize=(40,20))

    i=1

    for year in years: #[1975,1980,1985, 1990]: #
        #we cannot follow the cohort for max years; for 2016 we do not have enough data
        if year > (2015 - max_years):
            break

        ax5 = fig5.add_subplot(6, 6, i)
        i = i+1

        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1, 1, 1)

        #print("cohort: "+str(year))
        cohort = authorStartEndCareerData[authorStartEndCareerData["start_year"]==year]
        cohort_authors = cohort["author"].values
        #cohort_len.append({'expected_length' : len(cohort_authors), 'actual':[]})

        gini_over_years = pd.Series(data=0, index=years)
        cumnum_over_years = pd.DataFrame(data=0, index=years, 
                                         columns=["mean", "std", "mean_f", "std_f", "mean_m", "std_m", "mean_n", "std_n"])

        # extract num publications/citations for the cohort in all future years
        for y in range(year, max(years)+1):
            #print("following years: "+str(y))

            # get all the authors data for each year and filter based on the authors that we are interested in
            temp = data[data["year"]==y]
            temp = temp[temp["author"].isin(cohort_authors)]

            temp_count = len(temp[criterion].astype("float").values)
            
            if temp_count > 0:
                # gini per year based on cumulative num of publications/citations of all authors in this year
                gini_over_years.loc[y] = calculate.gini(temp[criterion].astype("float").values)
                #cohort_len[len(cohort_len)-1]['actual'].append(temp[criterion].values.shape[0])
                #print(temp[criterion].astype("float").values.)

                temp_male = temp[temp["gender"]=="m"]
                temp_female = temp[temp["gender"]=="f"]
                temp_none = temp[temp["gender"]=="none"]

                cumnum_over_years.loc[y] = [np.mean(temp[criterion].astype("float").values), 
                                               np.std(temp[criterion].astype("float").values),
                                               np.mean(temp_female[criterion].astype("float").values), 
                                               stats.sem(temp_female[criterion].astype("float").values),
                                               np.mean(temp_male[criterion].astype("float").values), 
                                               stats.sem(temp_male[criterion].astype("float").values),
                                               np.mean(temp_none[criterion].astype("float").values), 
                                               stats.sem(temp_none[criterion].astype("float").values)]
            else:
                gini_over_years.loc[y] = 0
                cumnum_over_years.loc[y] = [0, 0, 0, 0, 0, 0, 0, 0]

        gini_years_df = pd.DataFrame(gini_over_years.reset_index())
        gini_years_df.columns = ["year", "gini"]
        #gini_per_cohort[year] = gini_years_df

        gini_years = gini_years_df["year"].values
        gini_coefs= gini_years_df["gini"].values
        selected_gini_df = gini_years_df[(gini_years_df["year"] >= year) &  (gini_years_df["year"] < (year+max_years))]
        ax2.plot(cohort_duration, selected_gini_df["gini"])

        #["mean", "std", "mean_f", "std_f", "mean_m", "std_m", "mean_n", "std_n"])
        cumnum_years_df = pd.DataFrame(cumnum_over_years.reset_index())
        cumnum_years_df.columns = ["year", "mean", "std", "mean_f", "sem_f", "mean_m", "sem_m", "mean_n", "sem_n"]
        #cumnum_per_cohort[year] = cumnum_years_df

        selected_cumnum_df = cumnum_years_df[(cumnum_years_df["year"] >= year) &  (cumnum_years_df["year"] < (year+max_years))]
        ax3.errorbar(cohort_duration, selected_cumnum_df["mean"].values,  yerr=selected_cumnum_df["std"].values)


        ax4 = plot_gender_numcum(ax4, cohort_duration, selected_cumnum_df, year)
        ax4.set_title("Cohort start-year: "+str(year))  
        ax4.set_ylabel(criteria_display)
        ax4.legend()
        fig4.savefig("fig/"+criterion+"_gender_"+str(year)+".png")

        ax5 = plot_gender_numcum(ax5, cohort_duration, selected_cumnum_df, year)
        ax5.set_title("Cohort start-year: "+str(year))



    ax2.set_ylabel('Gini')
    ax2.set_title('Inequality of al cohorts over '+str(max_years)+' years')
    if len(years)<10:
        ax2.legend(years)  
    fig2.savefig("fig/"+criterion+"_gini.png")

    ax3.set_ylabel(criteria_display)
    ax3.set_title('Mean/Std of al cohorts over '+str(max_years)+' years')
    if len(years)<10:
        ax3.legend(years)  

    fig3.savefig("fig/"+criterion+".png")

    fig5.savefig("fig/"+criterion+"_gender.png")

    plt.show()
