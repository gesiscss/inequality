import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calculate
import scipy.stats as stats
from sklearn import linear_model
import math
import seaborn as sns
from matplotlib.font_manager import FontProperties
import random 
import matplotlib.cm as cm



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


def get_cohort_careerage_df(data, cohort_start_years, max_career_age, criterion):
    #returns a dataframe: cohort start year, career age, gender, distribution of values (num pub or cum num pub or num cit or cum num cit) 
  
    #gender can be all, f or m or none
    cohort_careerage_df = pd.DataFrame(columns=["cohort_start_year", "career_age", "criterion", "gender", "values"])
    #cohort_careerage_df.set_index(["cohort_start_year", "career_age", "criterion", "gender"])

    #print(data.head(3))
    for start_year in cohort_start_years: 
        cohort = data[data["start_year"]==start_year]
        #print(cohort.head(1))
        cohort_authors = cohort["author"].values
        cohort_size  = len(cohort_authors)
        
        # Problem: authors who do not publish in y year dont show up
        # we need to set their value to 0 or to the value of previous year (for cumulative calculation)   
        df_values = pd.DataFrame(cohort[['author', 'gender']])
        df_values['prev_value'] = [0.0]* df_values.shape[0]

        subsequent_years = [(start_year+i) for i in range(0, max_career_age)]
        
        # extract num publications/citations for the cohort in all future years
        age = 0
        for y in subsequent_years:
        
            age = age+1
            values = pd.Series(data=0) #index=range(0, cohort_size)
            
            temp = cohort[cohort["year"]==y]
            
            
            # make sure cohort is not shrinking
            df_values = pd.merge(df_values[['author', 'prev_value']],temp[['author','gender',criterion]], how='left', on='author')
      
            #Take the current values. If NaN or None then consider the previous values
            df_values[criterion] = df_values[criterion].combine_first(df_values['prev_value'])
        
      
            # If it is cumulative then previous values is set with current
            # Otherwise previous value will always be 0
            if(criterion.startswith('cum_')) :
                df_values['prev_value'] = df_values[criterion]

     
            
            all_values = df_values[criterion].astype("float").values
            #print("all_values for start_year:  "+str(start_year)+"  career_age: "+str(age)+" criterion: "+criterion+" gender: all" )
            #print(len(all_values))
            
            cohort_careerage_df = cohort_careerage_df.append({'cohort_start_year': start_year, 'career_age':age, 'criterion':criterion, 'gender':'all', 'values': all_values}, ignore_index=True)
                   
            
            temp_male = df_values[df_values["gender"]=="m"]
            temp_female = df_values[df_values["gender"]=="f"]
            temp_none = df_values[df_values["gender"]=="none"]
            
            cohort_careerage_df = cohort_careerage_df.append({'cohort_start_year': start_year, 'career_age':age, 'criterion':criterion, 'gender':'m', 'values': temp_male[criterion].astype("float").values}, ignore_index=True)
            cohort_careerage_df = cohort_careerage_df.append({'cohort_start_year': start_year, 'career_age':age, 'criterion':criterion, 'gender':'f', 'values': temp_female[criterion].astype("float").values}, ignore_index=True)
            cohort_careerage_df = cohort_careerage_df.append({'cohort_start_year': start_year, 'career_age':age, 'criterion':criterion, 'gender':'none', 'values': temp_none[criterion].astype("float").values}, ignore_index=True)

    
    
    return cohort_careerage_df
               
      
        
    
    
def get_cohort_stats(data, start_year, max_years, criterion, cohort_size_gini):
    # input: data frame, cohort start year, num years we want to follow each cohort
    # output: for the cohort that started in input-year we compute for each career age statistics
    # cumnum_over_years: compute mean, median, std of distribution (e.g. publications, citations cum pub, cum cit)
    # gini_over_years: stored for each career age of one cohort the gini of the distr (pub, cit, cum pub, cum cit)
    # cohort_size_gini: stores cohort start year, cohort size, career age and gini
    
    f = open('fig/inactive_'+criterion+'.txt','w') 


    cohort = data[data["start_year"]==start_year]
    cohort_authors = cohort["author"].values
    cohort_size  = len(cohort_authors)
    f.write("\n \n \n COHORT START YEAR: "+str(start_year)+"  ---  size:"+str(cohort_size)) 
    
    # Problem: authors who do not publish in y year dont show up
    # we need to set their value to 0 or to the value of previous year (for cumulative calculation)   
    df_values = pd.DataFrame(cohort[['author', 'gender']])
    df_values['prev_value'] = [0]* df_values.shape[0]

   
    
    subsequent_years = [(start_year+i) for i in range(0, max_years)]
    #subsequent_years = [(start_year+i) for i in np.arange(0, max_years,2)]
  
    
    gini_over_years = pd.Series(data=0, index=subsequent_years)
    cumnum_over_years = pd.DataFrame(data=0, index=subsequent_years, \
                                         columns=["mean", "std", "mean_f", "median_f", "std_f", "mean_m", "median_m", \
                                                  "std_m", "mean_n", "median_n", "std_n"])
    
    # extract num publications/citations for the cohort in all future years
    for y in subsequent_years:
       
        f.write("\n subsequent_year: "+str(y) )
        
        temp = cohort[cohort["year"]==y]
        #temp = cohort[(cohort["year"]==y) | (cohort["year"]==y+1)]
        
        # size will show how many people received citatins or published papers in that year
        #print("******************************** temp: cohort size in year"+str(y)+" size "+str(temp.shape[0]))
      
        # make sure cohort is not shrinking
        df_values = pd.merge(df_values,temp[['author',criterion]], how='left', on='author')
        #print("make sure cohort is not shrinking --> subsequent_year:  "+str(y)+" cohort values"+str(len(df_values[criterion].astype("float").values)))
        
        #Take the current values. If NaN or None then consider the previous values
        df_values[criterion] = df_values[criterion].combine_first(df_values['prev_value'])
        
      
        # If it is cumulative then previous values is set with current
        # Otherwise previous value will always be 0
        if(criterion.startswith('cum_')) :
            df_values['prev_value'] = df_values[criterion]

     
        temp_count = len(df_values[criterion].astype("float").values)
        #print("temp_count (length of distribution of pubs/cits):   "+str(temp_count))    
        
        if temp_count > 0:
            # gini per year based on (cum) num of publications/citations of all authors in this year    
            gini_over_years.loc[y] = calculate.gini(df_values[criterion].astype("float").values)
            
            # BUG
            # BUG: gini was previously computed based on temp dataframe which only contains values when people publish
            # Gini tends to increase with array-length and in more recent cohorts people publish more --> inequality emerges
            # if we compute gini over df_values we keep values from previous year is someone did not 
            # publish anything in the current year --> i.e. distribution becomes longer
            #gini_over_years.loc[y] = calculate.gini(temp[criterion].astype("float").values)
            
            #fraction of inactive people
            num_inactive_people = len(df_values[criterion].astype("float").values) - len(temp[criterion].astype("float").values)
            one_percent = len(df_values[criterion].astype("float").values)/100
            fraction_inactive = num_inactive_people/one_percent
            f.write(" fraction_inactive: "+str(fraction_inactive))
            
            
            temp_male = df_values[df_values["gender"]=="m"]
            temp_female = df_values[df_values["gender"]=="f"]
            temp_none = df_values[df_values["gender"]=="none"]

            cumnum_over_years.loc[y] = [np.mean(df_values[criterion].astype("float").values), 
                                               stats.sem(df_values[criterion].astype("float").values),
                                               np.mean(temp_female[criterion].astype("float").values), 
                                               np.median(temp_female[criterion].astype("float").values), 
                                               stats.sem(temp_female[criterion].astype("float").values),
                                               np.mean(temp_male[criterion].astype("float").values), 
                                               np.median(temp_male[criterion].astype("float").values), 
                                               stats.sem(temp_male[criterion].astype("float").values),
                                               np.mean(temp_none[criterion].astype("float").values), 
                                               np.median(temp_none[criterion].astype("float").values), 
                                               stats.sem(temp_none[criterion].astype("float").values)]
        else:
            print("Here no values for gini calculation found!!!!!!!!!!!!!!!!!!!")
            gini_over_years.loc[y] = np.nan
            cumnum_over_years.loc[y] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
               
        
        cohort_size_gini = cohort_size_gini.append( pd.DataFrame([[start_year, cohort_size, y, gini_over_years[y]]], 
                                 columns=["cohort_start_year", "cohort_size", "year", "gini"]), ignore_index =True)  
        # maintain only author and prev_value for calculations    
        df_values = df_values[['author','gender','prev_value']]
    
    f.close()     
    return (cumnum_over_years, cohort_size_gini, gini_over_years) 
    

def plot_cumulative_dist(df, age, criterion, criteria_display):
    # creates one cdf plot for each career age; each cohort is one line
    
    init_plotting()
    
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
        
        #df_one_age_one_cohort_values_unique = np.unique(df_one_age_one_cohort_values)
        
        #normalize values to make them compareable across cohort
        norm_values = (df_one_age_one_cohort_values-min(df_one_age_one_cohort_values))/(max(df_one_age_one_cohort_values)-min(df_one_age_one_cohort_values))
        
        i += 1
        plt.hist(norm_values, normed=True, cumulative=True, label='CDF', histtype='step', color=colors[i], linewidth=3)
    
    
    ax1.set_title('Career Age '+str(age))
    ax1.legend(cohort_start_years, loc=4)  
    fig1.savefig("fig/cdf_"+criterion+"_age"+str(age)+".png", facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')

        
 

        
def plot_cohort_analysis_on(data, criterion, cohort_start_years, max_years, criteria_display):
    
    # Compute and plot GINI of publication/citation distributions for each cohort over time.
    # criterion defines which distribution to use: publications, citations, cumulative publications, cumulative citations
    # Plots: 
    # (1) fig1: gini of (cumulative) number of publications/citations for each cohort over time,
    # (2) fig2: mean (cumulative) number of publications/citations for each cohort over time,
    # (3) fig3: mean (cumulative) number of publications/citations for each cohort over time,
    # (4) correlation cohort size and inequality --> plot_cohort_size_gini_cor
    # (5) correlation cohort start year and inequality --> plot_cohort_size_gini_cor
    
    # This analysis can be done year wise - for which we use cumulative values - inorder to avoid too many zeros
    # otherwise it can be done in multiples of years - In this case we don't use cumulative values but the absolute ones 
    # because the year duration is highly - most likely the authors will publish/get cited
    ########### But for our analysis, we have used year wise and cumulative values. 
    
    # data - the dataframe containing author publications or citations data
    # criterion - 'cum_num_pub' (or) 'cum_num_cit' (or) 'num_pub' (or) 'num_cit'
      # If you are referring to cumulative values then the name should start with 'cum_'
    # max_years - no. of years the analysis need to be carried out
    # criteria_display - 'Cumulative Publication Count' (or) 'Cumulative Citation Count'
    
    plt = init_plotting()
    
    #if years are grouped then get the step limit - infer the group
    step = cohort_start_years[1] - cohort_start_years[0]
    
    #store for each cohort and year gini of publication and citation distribution
    gini_per_cohort = pd.DataFrame(index=cohort_start_years)
    #store for each cohort and year gini of cumulative publication and citationdistribution
    cumnum_per_cohort = pd.DataFrame(index=cohort_start_years)

    #(1) fig1: gini of (cumulative) number of publications/citations for each cohort over time
    fig1 = plt.figure()
    #plt.ylim(0, 0.7)
    fig1.patch.set_facecolor('white')
    ax1 = fig1.add_subplot(1,1,1) #axisbg="white"
    
    
    #(2) fig2: mean (cumulative) number of publications/citations for each cohort over time,
    fig2 = plt.figure()
    fig2.patch.set_facecolor('white')
    ax2 = fig2.add_subplot(1,1,1) #, axisbg="white"

    # limit plot to the N years during which we follow a cohort
    cohort_duration = np.arange(max_years)
   
    i=0 # to point to the right figure
    j=0
    cohort_size_gini = pd.DataFrame(columns=["cohort_start_year", "cohort_size", "year", "gini"])

     # rearange subplots dynamically
    cols=2

    # 15 cohorts
    if(len(cohort_start_years)>10):
        cols=6
    
    nrows = math.ceil(float(len(cohort_start_years))/float(cols))
    nrows = int(nrows)
    
    # (3) fig3: mean (cumulative) number of publications/citations for each cohort over time
    fig3, ax3 = plt.subplots(nrows, cols, sharex=True, sharey='row', figsize=(16,10)) #sharey=True, 
    # Create a big subplot to created axis labels that scale with figure size
    ax_outside = fig3.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax_outside.set_xlabel('Career Age', labelpad=20, fontweight='bold') 
    ax_outside.set_ylabel('Mean '+criteria_display, labelpad=20, fontweight='bold')

    plt.tight_layout() 
   
    # For each cohort, iterate all their careers and calculate the inequality of productivity (publications)
    # and success (citations)
    
    for year in cohort_start_years: 
       
        #get the cohort names
        #print(year)
        #print(start_years)
        #print("get_cohort_stats")
        (cumnum_over_years, cohort_size_gini, gini_over_years) = get_cohort_stats(data, year, max_years, criterion, cohort_size_gini)
        
        #print("RETURNED FROM get_cohort_stats")
        #print(cumnum_over_years.head(n=2))
        #print(cohort_size_gini.head(n=2))
        #print(gini_over_years.head(n=2))
        
        
        gini_years_df = pd.DataFrame(gini_over_years.reset_index())
        gini_years_df.columns = ["year", "gini"]
        #gini_per_cohort[year] = gini_years_df

        #print(gini_years_df.head())
        
        gini_years = gini_years_df["year"].values
        gini_coefs= gini_years_df["gini"].values
        selected_gini_df = gini_years_df[(gini_years_df["year"] >= year) &  \
                                         (gini_years_df["year"] < (year+max_years))]
    
        
        # limit plot to the N years during which we follow a cohort
        #the length of cohort duration (x-axis) and y-axis values have to be the same
        # faced by some problems regarding that because of grouping and so, the x-axis count values are taken from the final table
        # say for e.g. when grouped by 2 years the values will be 0,2,4,6 ...
        cohort_duration = np.arange(0,len(selected_gini_df["gini"].values)*step, step)
        
        ax1.plot(cohort_duration, selected_gini_df["gini"])

        #["mean", "std", "mean_f", "std_f", "mean_m", "std_m", "mean_n", "std_n"])
        cumnum_years_df = pd.DataFrame(cumnum_over_years.reset_index())
        cumnum_years_df.columns = ["year", "mean", "std", "mean_f", "median_f", "sem_f", \
                                   "mean_m", "median_m", "sem_m", "mean_n", "median_n", "sem_n"]

        selected_cumnum_df = cumnum_years_df[(cumnum_years_df["year"] >= year) &  \
                                             (cumnum_years_df["year"] < (year+max_years))]
        ax2.errorbar(cohort_duration, selected_cumnum_df["mean"].values,  yerr=selected_cumnum_df["std"].values)
   
        
        ## plots the mean of publication/citation gender wise for each cohort
        ax3[i,j] = plot_gender_numcum(ax3[i,j], cohort_duration, selected_cumnum_df, year, "mean")
        ax3[i,j].set_title(str(year), fontsize=12, fontweight="bold")
   
        if (j<cols-1):
            j = j+1
        else:
            j=0
            i = i+1
 
    # save gini results for cohort
    cohort_size_gini.to_csv("fig/gini_"+criterion+"_results.csv")
    
    ## plots the correlation plot between gini and cohort size
    plot_cohort_size_gini_cor(cohort_size_gini, criterion, max_years, criteria_display)
    
    ax1.set_ylabel('Gini ('+criteria_display+')', fontweight='bold')
    ax1.set_xlabel('Career Age', fontweight='bold')

 
    if len(cohort_start_years)<10:
        ax1.legend(cohort_start_years)  
    fig1.savefig("fig/gini_"+criterion+".png", facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')

    ax2.set_ylabel("Mean "+criteria_display, fontweight='bold')
    ax2.set_xlabel('Career Age', fontweight='bold')
   

    if len(cohort_start_years)<10:
        ax2.legend(cohort_start_years)  

    fig2.savefig("fig/mean_"+criterion+".png", facecolor=fig2.get_facecolor(), edgecolor='none', bbox_inches='tight')
    fig3.savefig("fig/mean_"+criterion+"_gender.png", facecolor=fig3.get_facecolor(), edgecolor='none', bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    
    
def plot_cohort_size_gini_cor(data, criterion, max_years, criteria_display):
    
    # It computes the correlation between cohort size and gini for all cohorts at every career age
    # and plots it for each career age 
    
    # data - the dataframe contains: cohort-start-year, cohort-size, year, gini
    # criterion - 'cum_num_pub' (or) 'cum_num_cit' (or) 'num_pub' (or) 'num_cit'
      # If you are referring to cumulative values then the name should start with 'cum_'
    # max_years - no. of years the analysis need to be carried out
    # criteria_display - 'Cumulative Publication Count' (or) 'Cumulative Citation Count'
    
    #plt = init_plotting()
    
    res_cor_size = pd.DataFrame(columns=["career_age", "cor", "p", "num_obs"])
    res_cor_year = pd.DataFrame(columns=["career_age", "cor", "p", "num_obs"])
    
    # transfor subsequent year into 0-15
    data["career_age"] = data["year"]  - data["cohort_start_year"] 
    data["ordered_cohort_start_year"] = data["cohort_start_year"] - 1970
    #print(data.head())
    
    #fig = plt.figure(figsize=(16,10))
    #ax.set_xlabel("cohort size")
    #ax.set_ylabel("Gini")
    
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
    ax_outside.set_ylabel('Gini ('+criteria_display+')', labelpad=20, fontweight="bold")
    
    #(2) plot cor between cohort start year and gini
    fig2, ax2 = plt.subplots(nrows=nrows, ncols=cols, sharex=True, sharey=True, figsize=(16,10))
    ax_outside2 = fig2.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax_outside2.set_xlabel('Cohort Start Year', labelpad=20, fontweight="bold") 
    ax_outside2.set_ylabel('Gini ('+criteria_display+')', labelpad=20, fontweight="bold")

    
    #fig.text(0.5, 0.0, 'Cohort Size', ha='center')
    #fig.text(0.0, 0.5, 'Gini ('+criteria_display+')', va='center', rotation='vertical')
    
    plt.tight_layout()
       
    unique_career_ages = np.unique(data["career_age"])
    print(unique_career_ages)
    i=0
    j=0
    # get gini and cohort size data for each career age
    for age in unique_career_ages:
        temp = data[data["career_age"]==age]
        pcor = stats.pearsonr(temp["cohort_size"], temp["gini"])
        pcor2 = stats.pearsonr(temp["cohort_start_year"], temp["gini"])
        num_obs = len(temp["cohort_size"])
        num_obs = len(temp["cohort_start_year"])
        res_cor_size = res_cor_size.append(pd.DataFrame([[age, pcor[0], pcor[1], num_obs]], columns=["career_age", "cor", "p", "num_obs"]), 
                         ignore_index=True)
        res_cor_year = res_cor_year.append(pd.DataFrame([[age, pcor[0], pcor[1], num_obs]], columns=["career_age", "cor", "p", "num_obs"]), 
                         ignore_index=True)
        if age < max_years:    
            #ax1 = fig.add_subplot(nrows, cols, i)
            
            ax2[i,j].scatter(temp["cohort_start_year"], temp["gini"],  c="r", s=6)
            m, b = np.polyfit(temp["cohort_start_year"], temp["gini"], 1)
            ax2[i,j].plot(temp["cohort_start_year"], m*temp["cohort_start_year"] + b, '-')
            ax2[i,j].set_title("Age "+str(int(age))+" (c="+str(np.around(pcor2[0], decimals=2))+" p="+str(np.around(pcor2[1],decimals=3))+")", fontsize=12, fontweight="bold")
            #ax2[i,j].text(1975, 0.16, "cor="+str(np.around(pcor2[0], decimals=2))+" p="+str(np.around(pcor2[1],decimals=2)))
            labels =  ax2[i,j].get_xticklabels()
            plt.setp(labels,  rotation=45, fontsize=12, figure=fig2)

            ax[i,j].scatter(temp["cohort_size"], temp["gini"],  c="r", s=6)
            m, b = np.polyfit(temp["cohort_size"], temp["gini"], 1)
            ax[i,j].plot(temp["cohort_size"], m*temp["cohort_size"] + b, '-')
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
    
    
    
def plot_gender_numcum(ax, cohort_duration, selected_cumnum_df, year, selected_stat):
    init_plotting()
    ax.plot(cohort_duration, selected_cumnum_df[selected_stat+"_f"].values,  label='women', color="red")
    if (selected_stat == "mean"):
        ax.fill_between(cohort_duration, selected_cumnum_df["mean_f"].values-selected_cumnum_df["sem_f"].values, 
                    selected_cumnum_df["mean_f"].values+selected_cumnum_df["sem_f"].values,
					alpha=0.2, edgecolor='red', facecolor='red',
					linewidth=4, linestyle='dashdot', antialiased=True)
    
    ax.plot(cohort_duration, selected_cumnum_df[selected_stat+"_m"].values,  label='men', color="blue")
    if(selected_stat == "mean"):
        ax.fill_between(cohort_duration, selected_cumnum_df["mean_m"].values-selected_cumnum_df["sem_m"].values, 
                    selected_cumnum_df["mean_m"].values+selected_cumnum_df["sem_m"].values,
					alpha=0.2, edgecolor='blue', facecolor='blue',
					linewidth=4, linestyle='dashdot', antialiased=True)

    ax.plot(cohort_duration, selected_cumnum_df[selected_stat+"_n"].values,  label='unknown', color="grey")
    if(selected_stat == "mean"):
        ax.fill_between(cohort_duration, selected_cumnum_df["mean_n"].values-selected_cumnum_df["sem_n"].values, 
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
                                'residue':regr.residues_[0]}, \
                                ignore_index=True)
        
    # plot the regression analysis
    fig = plt.figure(figsize=(15,5))
    fig.suptitle('Regression Analysis of cohort performance on their inequality ')
    ax1 = fig.add_subplot(1,1,1)
    

    ax1.plot(analysis_output['year'],analysis_output['intercept'],label='Intercept')
    ax1.plot(analysis_output['year'],analysis_output['reg_coeff'],label='Reg. coefficient -b1')
    ax1.plot(analysis_output['year'],analysis_output['residue'],label='Residue')
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