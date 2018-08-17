import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calculate
from scipy import stats
from sklearn import linear_model
import math
import seaborn as sns
from matplotlib.font_manager import FontProperties
import random 
import matplotlib.cm as cm
from collections import Counter
from matplotlib.lines import Line2D

# set global settings
def init_plotting():
    #print(plt.style.available)
    #plt.style.use(['seaborn-paper'])
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams['font.size'] = 14
    plt.style.use(['seaborn-whitegrid'])
    #plt.rcParams['figure.figsize'] = (8, 3)
    #plt.gca().spines['right'].set_color('none')
    #plt.gca().spines['top'].set_color('none')
    #plt.gca().xaxis.set_ticks_position('bottom')
    #plt.gca().yaxis.set_ticks_position('left')
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    #plt.rcParams["font.weight"] = "bold"
    #plt.rcParams["axes.labelweight"] = "bold"
    #plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'upper left'
    plt.rcParams['axes.linewidth'] = 4
    return plt


def run_cohort_analysis(groupByYearData, cohort_start_years, max_career_age_cohort, criterion, criterion_display, authorStartEndCareerData):
    cohort_careerage_df = get_cohort_careerage_df(groupByYearData, cohort_start_years, max_career_age_cohort, criterion, authorStartEndCareerData)
    
    #gini
    cohort_size_gini = get_cohort_size_gini(cohort_careerage_df,criterion, cohort_careerage_df["cohort_start_year"].unique())
    #cohort_size_gini = get_cohort_gini(cohort_careerage_df,criterion, np.array([1970, 1980, 1990, 2000]))
    
    
    plot_gini(cohort_size_gini, criterion, criterion_display)
    plot_cohort_size_gini_cor(cohort_size_gini,  criterion, criterion_display)
    
    # mean/std/median
    stats = get_cohort_stats(cohort_careerage_df, criterion)
    print(stats.head(n=1))
    plot_cohort_gender_diffs(stats, criterion, criterion_display)
    plot_cohort_means_over_ages(stats, criterion, criterion_display)
    
    # plot overall gini for all authors that started between 1970 and 2000, ignore cohorts
    cohort_size_gini = get_cohort_size_gini(cohort_careerage_df,criterion, np.array([1970, 2000]))
    plot_gini(cohort_size_gini, criterion+"_ALL_AUTHORS", criterion_display)
    
    cohort_size_gini = get_cohort_size_gini(cohort_careerage_df,criterion, np.array([1970, 1980, 1990, 2000]))
    plot_gini(cohort_size_gini, criterion+"_COHORTS_10_YEARS", criterion_display)
    
    # plot effect size 
    cohort_effect_size = get_cohort_effect_size(cohort_careerage_df)
    plot_cohort_effect_size(cohort_effect_size)
    
    
def get_cohort_effect_size(cohort_careerage_df, group_a='m', group_b='f'):
    data = cohort_careerage_df[cohort_careerage_df.gender.isin([group_a, group_b])]
    return data.groupby(['cohort_start_year', 'career_age'])['values'].apply(lambda x: mann_whitney_effect_size(x.iloc[0],x.iloc[1], effect_formula='r'))
    
    
    
def get_cohort_careerage_df(data, cohort_start_years, max_career_age, criterion, authorStartEndCareerData):
    #returns a dataframe: cohort start year, career age, gender, distribution of values (num pub or cum num pub or num cit or cum num cit) 
  
    #gender can be all, f or m or none
    cohort_careerage_df = pd.DataFrame(columns=["cohort_start_year", "career_age", "criterion", "gender", "values"])

    for start_year in cohort_start_years: 
        cohort = data[data["start_year"]==start_year]
      
        cohort_authors = authorStartEndCareerData[authorStartEndCareerData.start_year == start_year]
        cohort_size  = cohort_authors.author.nunique()
        
        #print("\n \n \n COHORT START YEAR: "+str(start_year)+"  ---  size:"+str(cohort_size)) 
        subsequent_years = [(start_year+i) for i in range(0, max_career_age)]
        
        # extract num publications/citations for the cohort in all future years
        age = 1
        prev_value = [0.0] * cohort_size
        for y in subsequent_years:
            temp = cohort[cohort["year"]==y]
            active_people = temp["author"].nunique()
            
            # Problem: authors who do not publish in y year dont show up; only in first year everyone is active per definition!
            # we need to set their value to 0 or to the value of previous year (for cumulative calculation) 
            df_values = cohort_authors[['author', 'gender']].merge(temp[['author', criterion]], on='author', how='left')
            df_values['prev_value'] = prev_value
#             print(df_values.tail())

            # make sure cohort is not shrinking
            # df_values = pd.merge(df_values[['author', 'prev_value']],temp[['author','gender',criterion]], how='left', on='author')
            
            #Take the current values. If NaN or None then consider the previous values
            
            df_values[criterion] = df_values[criterion].combine_first(df_values['prev_value'])
            #print(df_values.tail())
      
            # If it is cumulative then previous values is set with current
            # Otherwise previous value will always be 0
            # 
            # this dataframe contains num_pub and cum_num_pub as row, but if criterion is cum_ then only cum_ col will be updated!
            #
            if(criterion.startswith('cum_')):
                prev_value = df_values[criterion]

            all_values = df_values[criterion].astype("float").values
            
          
            #fraction of inactive people
            #one_percent = cohort_size/100
            #fraction_inactive = (cohort_size - active_people)/one_percent
            #print(" fraction_inactive: "+str(fraction_inactive) +" in year "+str(y))
            #print("all_values for start_year:  "+str(start_year)+"  career_age: "+str(age)+" criterion: "+criterion+" gender: all, values "+ str(len(all_values)))
                   
                    
            cohort_careerage_df = cohort_careerage_df.append({'cohort_start_year': start_year, 'career_age':age, 'criterion':criterion, 'gender':'all', 'values': all_values}, ignore_index=True)
                   
            
            temp_male = df_values[df_values["gender"]=="m"]
            temp_female = df_values[df_values["gender"]=="f"]
            temp_none = df_values[df_values["gender"]=="none"]
            
            cohort_careerage_df = cohort_careerage_df.append({'cohort_start_year': start_year, 'career_age':age, 'criterion':criterion, 'gender':'m', 'values': temp_male[criterion].astype("float").values}, ignore_index=True)
            cohort_careerage_df = cohort_careerage_df.append({'cohort_start_year': start_year, 'career_age':age, 'criterion':criterion, 'gender':'f', 'values': temp_female[criterion].astype("float").values}, ignore_index=True)
            cohort_careerage_df = cohort_careerage_df.append({'cohort_start_year': start_year, 'career_age':age, 'criterion':criterion, 'gender':'none', 'values': temp_none[criterion].astype("float").values}, ignore_index=True)
            
            age = age+1
        
        
       
            # test a guy(girl) that is in cohort 2001
            # 'maseka lesaoana published 2 publications in total, one in 2001 and one in 2015
            # every year she got some citations
            #if start_year == 2001:
            #    print("+++++++++++++++ cohort 2001 +++++++++++++++++++++++++")
            #    print(df_values[df_values["author"] == '\'maseka lesaoana'])
 
    return cohort_careerage_df
               
      
# get gini for each year or intervals longer than a year        
def get_cohort_size_gini(data, criterion, start_years):
    # input dataframe: cohort start year, career age, gender, distribution of values (num pub or cum num pub or num cit or cum num cit) 
    # cohort_size_gini: stores cohort start year, cohort size, career age and gini
       
    cohort_size_gini = pd.DataFrame(columns=["cohort_start_year", "cohort_size", "age", "gini"])
          
    #for start_year in start_years:
    i=0
    while i in range(0, (len(start_years)-1)):
        start_year_start = start_years[i]
        start_year_end =  start_years[i+1] 
        i = i+1
        #print("start_year_start: "+str(start_year_start)+"  start_year_end: "+str(start_year_end))
        
        #cohort = data[data["cohort_start_year"] == start_year]
        data = data[data["cohort_start_year"] >= start_year_start]
        cohort = data[data["cohort_start_year"] < start_year_end]
        
        cohort = cohort[cohort["criterion"] == criterion]
        
        for age in np.unique(cohort["career_age"]):
            cohort_age_df = cohort[cohort["career_age"] == age]
            
            #gini can only be computed for full dataset
            cohort_age_df = cohort_age_df[cohort_age_df["gender"] == "all"]
            
            #print(len(cohort_age_df["values"].values[0]))
            values = cohort_age_df["values"].values[0]
            
            if(len(values)>0):
                gini = calculate.gini(values)
                cohort_size_gini = cohort_size_gini.append({'cohort_start_year': start_year_start, 'cohort_size':len(values), 'age': age, 'gini': gini}, ignore_index=True) 
            else:
                print("Here no values for gini calculation found!!!!!!!!!!!!!!!!!!!")
                cohort_size_gini = cohort_size_gini.append({'cohort_start_year': start_year_start, 'cohort_size':len(values), 'age': age, 'gini': np.nan}, ignore_index=True)       
    
    
    # save gini results for cohort
    cohort_size_gini.to_csv("fig/gini_"+criterion+".csv")
    
    return cohort_size_gini    

                                                       

def get_cohort_stats(data, criterion):
    # input dataframe: cohort start year, career age, gender, distribution of values (num pub or cum num pub or num cit or cum num cit) 
    # output:  mean, median, std of distribution (e.g. publications, citations cum pub, cum cit)
   
    stats = pd.DataFrame(columns=["cohort_start_year", "cohort_size", "age", "criterion", "mean", "std", "sem", "median", "mean_f", "median_f", "std_f", "sem_f", "mean_m", "median_m", "std_m", "sem_m", "mean_n", "median_n", "std_n", "sem_n"])
    
      
    for start_year in data["cohort_start_year"].unique():
        cohort = data[data["cohort_start_year"] == start_year]
        cohort = cohort[cohort["criterion"] == criterion]
        
        for age in np.unique(cohort["career_age"]):
            cohort_age_df = cohort[cohort["career_age"] == age]
           
                                                       
            cohort_age_all = cohort_age_df[cohort_age_df["gender"] == "all"]
            cohort_age_male = cohort_age_df[cohort_age_df["gender"] == "m"]
            cohort_age_female = cohort_age_df[cohort_age_df["gender"] == "f"]
            cohort_age_none = cohort_age_df[cohort_age_df["gender"] == "none"]
            
            
            values_all = cohort_age_all["values"].values[0]
            values_male = cohort_age_male["values"].values[0]
            values_female = cohort_age_female["values"].values[0]
            values_none = cohort_age_none["values"].values[0]
            
       
            stats = stats.append({'cohort_start_year': start_year, 'cohort_size':len(values_all), 'age': age, 'criterion':criterion,
                                  'mean': np.mean(values_all), 'std':np.std(values_all), 'sem':np.std(values_all)/np.sqrt(len(values_all)),
                                  'median':np.median(values_all),
                                  'mean_f': np.mean(values_female), 'std_f':np.std(values_female),
                                  'sem_f':np.std(values_female)/np.sqrt(len(values_female)), 'median_f':np.median(values_female),
                                  'mean_m': np.mean(values_male), 'std_m':np.std(values_male),
                                  'sem_m':np.std(values_male)/np.sqrt(len(values_male)),'median_m':np.median(values_male),
                                  'mean_n': np.mean(values_none), 'std_n':np.std(values_none),
                                  'sem_n':np.std(values_none)/np.sqrt(len(values_none)),'median_n':np.median(values_none)},
                                 ignore_index=True) 
        
    
  
    stats.to_csv("fig/stats_"+criterion+".csv")
   
    return stats
    

    

def plot_cumulative_dist(df, age, criterion, criteria_display):
    # creates one cdf plot for each career age; each cohort is one line
    
    plt = init_plotting()
    
    fig1 = plt.figure()
    fig1.patch.set_facecolor('white')
    ax1 = fig1.add_subplot(1,1,1) #axisbg="white"
    

    df_one_age = df[df["career_age"]==age]
    df_one_age = df_one_age[df_one_age["criterion"]==criterion]
    df_one_age = df_one_age[df_one_age["gender"]=='all']
 
    
    cohort_start_years = np.unique(df_one_age["cohort_start_year"].values)
    numcolors = len(cohort_start_years)+1

    colors = cm.rainbow(np.linspace(0, 1, numcolors))                                
    i = 0
                        
    for y in cohort_start_years:
   
        df_one_age_one_cohort = df_one_age[df_one_age["cohort_start_year"]==y]
        arr = df_one_age_one_cohort["values"].values
        
        # there should be only one array per cohort-start-year and career age
        #for item in arr:
        #    print("****** YEAR: "+str(y)+"  ---  "+str(len(item)))
            
        df_one_age_one_cohort_values = np.sort(df_one_age_one_cohort["values"].values[0])
        cohort_values_frequency = Counter(df_one_age_one_cohort_values)
        print("  year:"+str(y)+"    age: "+str(age))
        print(cohort_values_frequency)
                    
        #careerLengthDist = df_one_age_one_cohort_values.groupby(["career_length"])['author'].count()
        #temp = careerLengthDist.cumsum()  
        #ax = temp.plot(grid=True, title='Cumulative histogram of Career Length by authors')
        #ax.set_xlabel('Career Length')
        #ax.set_ylabel('No. of Authors')
        #df_one_age_one_cohort_values_unique = np.unique(df_one_age_one_cohort_values)
        
        #normalize values to make them compareable across cohort
        norm_values = (df_one_age_one_cohort_values-min(df_one_age_one_cohort_values))/(max(df_one_age_one_cohort_values)-min(df_one_age_one_cohort_values))
        
        i += 1
        plt.hist(norm_values, normed=True, cumulative=True, label='CDF', histtype='step', color=colors[i], linewidth=3)
    
    
    ax1.set_title('Career Age '+str(age))
    ax1.legend(cohort_start_years, loc=4)  
    fig1.savefig("fig/cdf_"+criterion+"_age"+str(age)+".png", facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')

         
def plot_gini(cohort_size_gini, criterion, criteria_display):
    #input  dataframe ["cohort_start_year", "cohort_size", "age", "gini"])   
    plt = init_plotting()
    
    fig1 = plt.figure()
    #plt.ylim(0.1, 1)
    
    fig1.patch.set_facecolor('white')
    ax1 = fig1.add_subplot(1,1,1) #axisbg="white"
    cohort_start_years = np.unique(cohort_size_gini["cohort_start_year"].values)
    
    for start_year in cohort_start_years:
        selected_cohort = cohort_size_gini[cohort_size_gini["cohort_start_year"]==start_year]

        ax1.plot(selected_cohort["age"], selected_cohort["gini"])
        
    ax1.set_ylabel('Gini ('+criteria_display+')', fontweight='bold')
    ax1.set_xlabel('Career Age', fontweight='bold')
    
    if ((len(cohort_start_years)<10) & (len(cohort_start_years)>1)):
        ax1.legend(cohort_start_years)  
    plt.show()
    fig1.savefig("fig/gini_"+criterion+".png", facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig1)

    
def plot_cohort_gender_diffs(data, criterion, criteria_display):
   
    plt = init_plotting()

    fig2 = plt.figure()
    fig2.patch.set_facecolor('white')
    ax2 = fig2.add_subplot(1,1,1) #, axisbg="white"

    cohort_start_years = np.unique(data["cohort_start_year"].values)
    
   
    highlighted_cohorts = []  
    colors = ('#DE4C2C', '#3BD64C', '#3B9ED6', '#B73BD6', '#F39C12', '#FFC0CB', '#27AE60', '#48C9B0', '#071019') #'#AAB7B8',
    markers = []
    for m in Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass
 
    for year in cohort_start_years: 
        
        cohort = data[data["cohort_start_year"]==year]
        cohort = cohort[cohort["criterion"]==criterion]   
#         print(cohort.head())
        # print(calculate.cohen_d(cohort["mean_m"].values, cohort["mean_f"].values, cohort["std_m"].values, cohort["std_f"].values))
        # buggs in calculate.cohen_d
       
    
def plot_cohort_means_over_ages(data, criterion, criteria_display):
    # Plots: 
    # (2) fig2: mean (cumulative) number of publications/citations for - single plot,
    # x - career age
    # highlighting certain cohorts
    
    # (3) fig3: mean (cumulative) number of publications/citations for each cohort over career ages - grid of plots,
    # x - career age
    # colors indicate gender (m/f/none)
 
    
    plt = init_plotting()

    #(2) fig2
    # ==============================================#
    fig2 = plt.figure()
    fig2.patch.set_facecolor('white')
    ax2 = fig2.add_subplot(1,1,1) #, axisbg="white"
    
    
    highlighted_cohorts = []  
    colors = ('#DE4C2C', '#3BD64C', '#3B9ED6', '#B73BD6', '#F39C12', '#FFC0CB', '#27AE60', '#48C9B0', '#071019') #'#AAB7B8',
    markers = []
    for m in Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            print("Typeerror occured")
            pass
    
    
    #(3) fig3
    # ==============================================#
    i=0 # to point to the right figure
    j=0
    
    # rearange subplots dynamically
    cols=2
    cohort_start_years = np.unique(data["cohort_start_year"].values)
    
    # 15 cohorts
    if(len(cohort_start_years)>10):
        cols=6
    
    nrows = math.ceil(float(len(cohort_start_years))/float(cols))
    nrows = int(nrows)

    fig3, ax3 = plt.subplots(nrows, cols, sharex=True, sharey='row', figsize=(16,10)) #sharey=True, 
    # Create a big subplot to created axis labels that scale with figure size
    ax_outside = fig3.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax_outside.set_xlabel('Career Age', labelpad=20, fontweight='bold') 
    ax_outside.set_ylabel('Mean '+criteria_display, labelpad=20, fontweight='bold')

    p=0
    for year in cohort_start_years: 
        
        cohort = data[data["cohort_start_year"]==year]
        cohort = cohort[cohort["criterion"]==criterion]   
       
        # compute cliff d between  distr of means over career ages
        cliffd_m_f = calculate.cliffsD(cohort["mean_m"].values, cohort["mean_f"].values)
      
        #cumnum_over_years = get_cohort_stats(df, year, max_years, criterion)
        #cohort_duration = np.arange(0,len(selected_gini_df["gini"].values)*step, step)
        
        #(2) fig2
        # ==============================================#
        #show some selected years in color and show legend for the
        if(year%5 == 0):
            ax2.errorbar(cohort["age"], cohort["mean"].values,  yerr=cohort["sem"].values, label=year, color=colors[p],
                         marker=markers[p], markersize=10)
            highlighted_cohorts.append(year)
            p = p+1 
        else:
            ax2.errorbar(cohort["age"], cohort["mean"].values,  yerr=cohort["sem"].values, color='grey', alpha=0.5)
    
        #(3) fig3
        # ==============================================# 
        ## plots the mean of publication/citation gender wise for each cohort
        ax3[i,j] = plot_gender_numcum(ax3[i,j], cohort["age"], cohort, "mean")
        
        ax3[i,j].set_title(str(year)+"("+str(np.round(cliffd_m_f))+")", fontsize=12, fontweight="bold")
   
        if (j<cols-1):
            j = j+1
        else:
            j=0
            i = i+1
 
    ax2.set_ylabel("Mean "+criteria_display, fontweight='bold')
    ax2.set_xlabel('Career Age', fontweight='bold')
   
    ax2.legend()
    #if len(cohort_start_years)<10:
    #    ax2.legend(cohort_start_years)  
    #else:
    #    ax2.legend(highlighted_cohorts)

    fig2.savefig("fig/mean_"+criterion+".png", facecolor=fig2.get_facecolor(), edgecolor='none', bbox_inches='tight')
    fig3.savefig("fig/mean_"+criterion+"_gender.png", facecolor=fig3.get_facecolor(), edgecolor='none', bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig2)
    plt.close(fig3)
    
    
def plot_cohort_size_gini_cor(data, criterion, criteria_display):
  
    # It computes the correlation between cohort size and gini for all cohorts at every career age
    # and plots it for each career age 
    # data - the dataframe contains: cohort-start-year, cohort-size, age, gini
    # criterion - 'cum_num_pub' (or) 'cum_num_cit' (or) 'num_pub' (or) 'num_cit'
  
    plt = init_plotting()
    
    res_cor_size = pd.DataFrame(columns=["career_age", "cor", "p", "num_obs"])
    res_cor_year = pd.DataFrame(columns=["career_age", "cor", "p", "num_obs"])
    
    unique_career_ages = np.unique(data["age"])
    max_years = np.max(unique_career_ages)
    print("plot_cohort_size_gini_cor ---  unique_career_ages:")
    print(unique_career_ages)
    
    if(max_years >= 15):
        cols = 5
    else:
        cols = 3   
    nrows = int(math.ceil(float(max_years)/float(cols)))
    
    # (1) plot cor between cohort size and gini
    fig, ax = plt.subplots(nrows=nrows, ncols=cols, sharex=True, sharey=True, figsize=(16,10))
    ax_outside = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax_outside.set_xlabel('Cohort Size', labelpad=20, fontweight="bold") 
    ax_outside.set_ylabel('Gini '+criteria_display+'', labelpad=20, fontweight="bold")
    
    #(2) plot cor between cohort start year and gini
    fig2, ax2 = plt.subplots(nrows=nrows, ncols=cols, sharex=True, sharey=True, figsize=(16,10))
    ax_outside2 = fig2.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax_outside2.set_xlabel('Cohort Start Year', labelpad=20, fontweight="bold") 
    ax_outside2.set_ylabel('Gini ('+criteria_display+')', labelpad=20, fontweight="bold")

    
    #fig.text(0.5, 0.0, 'Cohort Size', ha='center')
    #fig.text(0.0, 0.5, 'Gini ('+criteria_display+')', va='center', rotation='vertical')
    
    #plt.tight_layout()
       
   
    i=0
    j=0
    # get gini and cohort size data for each career age
    for age in unique_career_ages:
        temp = data[data["age"]==age]
        pcor = stats.pearsonr(temp["cohort_size"], temp["gini"])
        pcor2 = stats.pearsonr(temp["cohort_start_year"], temp["gini"])
        num_obs = len(temp["cohort_size"])
        num_obs = len(temp["cohort_start_year"])
        res_cor_size = res_cor_size.append(pd.DataFrame([[age, pcor[0], pcor[1], num_obs]], columns=["career_age", "cor", "p", "num_obs"]), 
                         ignore_index=True)
        res_cor_year = res_cor_year.append(pd.DataFrame([[age, pcor[0], pcor[1], num_obs]], columns=["career_age", "cor", "p", "num_obs"]), 
                         ignore_index=True)
         
        eps = np.finfo(np.float32).eps 
        ax2[i,j].scatter(temp["cohort_start_year"], temp["gini"],  c="b", s=6)
        m, b = np.polyfit(np.float32(temp["cohort_start_year"]), temp["gini"], 1, rcond = len(temp["cohort_start_year"])*eps)
        ax2[i,j].plot(temp["cohort_start_year"], m*temp["cohort_start_year"] + b, '-', c="C7")
        ax2[i,j].set_title("Age "+str(int(age))+" (c="+str(np.around(pcor2[0], decimals=2))+" p="+str(np.around(pcor2[1],decimals=3))+")", fontsize=12, fontweight="bold")
        #ax2[i,j].text(1975, 0.16, "cor="+str(np.around(pcor2[0], decimals=2))+" p="+str(np.around(pcor2[1],decimals=2)))
        labels =  ax2[i,j].get_xticklabels()
        plt.setp(labels,  rotation=45, fontsize=12, figure=fig2)
        

        ax[i,j].scatter(temp["cohort_size"], temp["gini"],  c="b", s=6)
        m, b = np.polyfit(np.float32(temp["cohort_size"]), temp["gini"], 1, rcond = len(temp["cohort_start_year"])*eps)
        ax[i,j].plot(temp["cohort_size"], m*temp["cohort_size"] + b, '-', c='C7')
        ax[i,j].set_title("Age "+str(int(age))+" (c="+str(np.around(pcor[0], decimals=2))+" p="+str(np.around(pcor[1],decimals=3))+")", fontsize=12, fontweight="bold")
        #ax[i,j].text(0.002, 0.16, "cor="+str(np.around(pcor[0], decimals=2))+" p="+str(np.around(pcor[1],decimals=2)))
        labels =  ax[i,j].get_xticklabels()
        plt.setp(labels, rotation=45, fontsize=12, figure=fig)
            
        if (j < cols-1):
            j = j+1
        else:
            j=0
            i = i+1
            
        
    res_cor_size.to_csv("fig/cor_cohortSize_gini_"+criterion+".csv")
    fig.show()
    fig.savefig("fig/cor_cohortSize_gini_"+criterion+".png", facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    
    res_cor_year.to_csv("fig/cor_cohortStartYear_gini_"+criterion+".csv")
    fig2.show()
    fig2.savefig("fig/cor_cohortStartYear_gini_"+criterion+".png", facecolor=fig2.get_facecolor(), edgecolor='none', bbox_inches='tight')
    
    #plt.close(fig)
    
    
    
def plot_gender_numcum(ax, cohort_duration, selected_cumnum_df, selected_stat):
    
    ax.plot(cohort_duration, selected_cumnum_df[selected_stat+"_f"].values,  label='women', color="red")
    if (selected_stat == "mean"):
        ax.fill_between(np.array(cohort_duration, dtype=float), selected_cumnum_df["mean_f"].values-selected_cumnum_df["sem_f"].values, 
                    selected_cumnum_df["mean_f"].values+selected_cumnum_df["sem_f"].values,
					alpha=0.2, edgecolor='red', facecolor='red',
					linewidth=4, linestyle='dashdot', antialiased=True)
    
    ax.plot(cohort_duration, selected_cumnum_df[selected_stat+"_m"].values,  label='men', color="blue")
    if(selected_stat == "mean"):
        ax.fill_between(np.array(cohort_duration, dtype=float), selected_cumnum_df["mean_m"].values-selected_cumnum_df["sem_m"].values, 
                    selected_cumnum_df["mean_m"].values+selected_cumnum_df["sem_m"].values,
					alpha=0.2, edgecolor='blue', facecolor='blue',
					linewidth=4, linestyle='dashdot', antialiased=True)

    ax.plot(cohort_duration, selected_cumnum_df[selected_stat+"_n"].values,  label='unknown', color="grey")
    if(selected_stat == "mean"):
        ax.fill_between(np.array(cohort_duration, dtype=float), selected_cumnum_df["mean_n"].values-selected_cumnum_df["sem_n"].values, 
                    selected_cumnum_df["mean_n"].values+selected_cumnum_df["sem_n"].values,
					alpha=0.2, edgecolor='grey', facecolor='grey',
					linewidth=4, linestyle='dashdot', antialiased=True) 
    
    
    return ax
	
	


    
def plot_regress_performance_on_inequality(data, criterion, years, max_years):
    
    # data - the dataframe containing author publications or citations data
    # criterion - 'cum_num_pub' (or) 'cum_num_cit' (or) 'num_pub' (or) 'num_cit'
    # max_years - no. of years the analysis need to be carried out
    
    step = years[1] - years[0]
    #print(step)
    
    analysis_output = pd.DataFrame(columns=['year','intercept','reg_coeff','residue'])
    
    # For each cohort, go through all their careers years and calculate the mean and GINI
    for year in years: 
        #we cannot follow the cohort for max years; for 2016 we do not have enough data
        if year > (END_YEAR - max_years):
            #print('I am breaking - ', year)
            break
            
        cohort = data[data["start_year"]==year]
        cohort_authors = cohort["author"].values
        
        #It is used to maintain previous values in cumulative calculations otherwise 0
        df_values = pd.DataFrame(cohort[['author']])
        df_values['prev_value'] = [0]* df_values.shape[0]

        gini_mean_per_cohort = pd.DataFrame(columns=['x','y'])
        
        subsequent_years = [yr for yr in years if yr >= year]
        
        # extract num publications/citations for the cohort in all future years
        for y in subsequent_years:
            
            # get all the authors data for each year and filter based on the authors that we are interested in
            temp = data[data["year"]==y]
            temp = temp[temp["author"].isin(cohort_authors)]
            
            df_values = pd.merge(df_values,temp[['author',criterion]],how='left',on='author')
            
            #Take the current values. If NaN or None then consider the previous values
            df_values[criterion] = df_values[criterion].combine_first(df_values['prev_value'])

            # If it is cumulative then previous values is set with current
            # Otherwise previous value will always will be 0
            if(criterion.startswith('cum_')) :
                df_values['prev_value'] = df_values[criterion]

            temp_count = len(df_values[criterion].astype("float").values)
         
            if temp_count > 0:
                gini = calculate.gini(df_values[criterion].astype("float").values.flatten())
                mean = np.mean(df_values[criterion].astype("float").values)
            else:
                gini = 0
                mean = 0
                
            gini_mean_per_cohort = gini_mean_per_cohort.append({'x':mean,'y':gini}, ignore_index=True)
            
            #print(df_values)
            # maintain only author and prev_value for calculations    
            df_values = df_values[['author','prev_value']]
                           
        # Create linear regression object and train it
        regr = linear_model.LinearRegression()
        regr.fit(gini_mean_per_cohort['x'].to_frame(),gini_mean_per_cohort['y'].to_frame())
        
        #store the results - intercept, regression coefficient and error
        analysis_output = analysis_output.append({'year':year, \
                                'intercept':regr.intercept_[0], \
                                'reg_coeff':regr.coef_[0][0], \
                                #'residue':regr.residues_[0]
                                 },ignore_index=True)
        
    # plot the regression analysis
    fig = plt.figure(figsize=(15,5))
    fig.suptitle('Regression Analysis of cohort performance on their inequality ')
    ax1 = fig.add_subplot(1,1,1)
    

    ax1.plot(analysis_output['year'],analysis_output['intercept'],label='Intercept')
    ax1.plot(analysis_output['year'],analysis_output['reg_coeff'],label='Reg. coefficient -b1')
    #ax1.plot(analysis_output['year'],analysis_output['residue'],label='Residue')
    ax1.set_xlabel('Year that cohort started')
    ax1.legend()

    fig.savefig("fig/regress_"+criterion+"_gini.png")
    plt.show()
    
    # return data for further analysis
    return analysis_output



def plot_cohort_participation_year_wise_for(authorScientificYearStartEnd, data, CAREER_LENGTH_LIST):
    
    # This produces a range of plots for each career specified in CAREER_LENGTH_LIST. Each plot will compare the fraction/number
    # of authors for each year cohort wise
    # one set of plots will refer to the fraction of authors whose career span is greater than specified and 
    # the other will specify the absolute number
    
    # authorScientificYearStartEnd - author start year, end year, publication count details
    # data - the dataframe containing author publications or citations data
    # CAREER_LENGTH_LIST - this is used to filter data based on their career span
    
    # infer the grouping from data itself and if so calculate the step
    group_years = data['year'].unique()
    group_years = sorted(group_years)
    
    step = group_years[1] - group_years[0]
    
    fig = plt.figure(figsize=(20,20))
    fig.suptitle('Evolution of cohort participation per '+str(step)+' year wise')
    fig2 = plt.figure(figsize=(20,20))
    fig2.suptitle('Evolution of cohort participation per '+str(step)+' year wise')

    i = 1
    for CAREER_LENGTH in CAREER_LENGTH_LIST:
        ax = fig.add_subplot(3,3,i)
        ax2 = fig2.add_subplot(3,3,i)
        i += 1
        # Two Dataframes - one to store quantity and the other fraction
        careerLengthSelectionFrame = pd.DataFrame(index=range(1,51,step))
        careerLengthSelectionFrameInNos = pd.DataFrame(index=range(1,51,step))

        for year in group_years:
            # Initially set everything to 0
            careerLengthSelectionFrame[year] = 0.0
            careerLengthSelectionFrameInNos[year] = 0
            # Get the list of authors who started at one year (i.e. Cohort) and filter them based on their career length
            cohort_authors = authorScientificYearStartEnd[authorScientificYearStartEnd['start_year'] == year]
            cohort_authors_with_credibility = cohort_authors[cohort_authors['career_length'] >= CAREER_LENGTH]

            #cohort_authors = cohort_authors['author'].values
            cohort_authors_with_credibility = cohort_authors_with_credibility['author'].values

            len_authors = float(len(cohort_authors_with_credibility))

            subsequent_years = [yr for yr in group_years if yr >= year]
            #ca - career age
            ca = 1
            if len_authors > 0:
                # For that cohort - go through their career and see how many of them have published at every year
                
                for sub_yr in  subsequent_years:
                    temp = data[data['year'] == sub_yr]
                    #temp = temp[temp['author'].isin(cohort_authors_with_credibility)]
                    temp = set(temp['author'].values)
                    temp = temp & set(cohort_authors_with_credibility)
                    careerLengthSelectionFrame.loc[ca][year]= float(len(temp))/len_authors
                    careerLengthSelectionFrameInNos.loc[ca][year]= len(temp)
                    ca += step
                    #print(len(temp))
                    #break
                #break
            else:
                for sub_yr in  subsequent_years:
                    careerLengthSelectionFrame.loc[ca][year]= 0.0
                    careerLengthSelectionFrameInNos.loc[ca][year]= 0
                    ca += step

        y_axis_label = 'Fraction of authors with Career length >= '+str(CAREER_LENGTH)
        ax.plot(careerLengthSelectionFrame)
        ax.set_xlabel('Career Age')
        ax.set_ylabel(y_axis_label)

        y_axis_label = 'No. of authors with Career length >= '+str(CAREER_LENGTH)
        ax2.plot(careerLengthSelectionFrameInNos)
        ax2.set_xlabel('Career Age')
        ax2.set_ylabel(y_axis_label)
        ax2.set_yscale('log')

    plt.show()
    
    
def plot_cliffs_delta(cliffsD_cohorts, p1, p2, p3):
    x = cliffsD_cohorts['start_year']
    y = cliffsD_cohorts['cliffsD']
    plt.scatter(x, y)
    fit = np.polyfit(x, y, deg=1)
    plt.plot(x, fit[0] * x + fit[1], color='C1')
    plt.plot(x, [0]*len(x), color='C2')
    plt.title("Cliff's Delta for "+p1+" and "+p2+" for "+p3)
    plt.xlabel('Cohort start year')
    plt.ylabel("Cliff's Delta")
    plt.show()