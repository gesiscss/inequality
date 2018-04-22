import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calculate
import scipy.stats as stats
from sklearn import linear_model
import math
import seaborn as sns
from matplotlib.font_manager import FontProperties

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

    

def plot_cohort_analysis_on(data, criterion, years, max_years, criteria_display):
    
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
    step = years[1] - years[0]
    
    #store for each cohort and year gini of publication and citation distribution
    gini_per_cohort = pd.DataFrame(index=years)
    #store for each cohort and year gini of cumulative publication and citationdistribution
    cumnum_per_cohort = pd.DataFrame(index=years)

    #(1) gini of (cumulative) number of publications/citations for each cohort over time
    fig1 = plt.figure()
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
    
    # if we use 2015 we have exactly 30 cohorts which nicely fits on a 5x6 plot
    cohort_start_years = [y for y in years if y < (2015 - max_years)]
    print(cohort_start_years)
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
        cohort = data[data["start_year"]==year]
        cohort_authors = cohort["author"].values
        cohort_size  = len(cohort_authors)
       
        #Maintain previous values in cumulative calculations otherwise 0
        df_values = pd.DataFrame(cohort[['author', 'gender']])
        df_values['prev_value'] = [0]* df_values.shape[0]

        gini_over_years = pd.Series(data=0, index=years)
        cumnum_over_years = pd.DataFrame(data=0, index=years, \
                                         columns=["mean", "std", "mean_f", "median_f", "std_f", "mean_m", "median_m", \
                                                  "std_m", "mean_n", "median_n", "std_n"])

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
            # Otherwise previous value will always be 0
            if(criterion.startswith('cum_')) :
                df_values['prev_value'] = df_values[criterion]

            temp_count = len(df_values[criterion].astype("float").values)
            
            if temp_count > 0:
                # gini per year based on cumulative num of publications/citations of all authors in this year
                gini_over_years.loc[y] = calculate.gini(temp[criterion].astype("float").values)
               
                      #cohort_len[len(cohort_len)-1]['actual'].append(temp[criterion].values.shape[0])
                #print(temp[criterion].astype("float").values.)

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
                # If we use cumulative values then this should not happen.
                # If publication and citation counts are used then it can happen that at one year for one cohort, depending
                # on the credibility chosen none of the authors might have published or got citation
                # In those cases, we subsitute GINI with NaN
                
                print("Here no values for gini calculation found!!!!!!!!!!!!!!!!!!!")
                gini_over_years.loc[y] = np.nan
                cumnum_over_years.loc[y] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
               
        
            cohort_size_gini = cohort_size_gini.append( pd.DataFrame([[year, cohort_size, y, gini_over_years[y]]], 
                                 columns=["cohort_start_year", "cohort_size", "year", "gini"]), ignore_index =True)  
            # maintain only author and prev_value for calculations    
            df_values = df_values[['author','gender','prev_value']]
            
        gini_years_df = pd.DataFrame(gini_over_years.reset_index())
        gini_years_df.columns = ["year", "gini"]
     
        #gini_per_cohort[year] = gini_years_df

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
        #ax2.fill_between(cohort_duration, selected_cumnum_df["mean"].values-selected_cumnum_df["std"].values, 
        #            selected_cumnum_df["mean"].values+selected_cumnum_df["std"].values,
	#				alpha=0.2, 
	#				linewidth=4, linestyle='dashdot', antialiased=True) 
       
        
        ## plots the mean of publication/citation gender wise for each cohort
        ax3[i,j] = plot_gender_numcum(ax3[i,j], cohort_duration, selected_cumnum_df, year, "mean")
        ax3[i,j].set_title(str(year), fontsize=12, fontweight="bold")
        #font0 = FontProperties()
        #font = font0.copy()
        #font.set_weight('bold')
        #ax3[i,j].text(9, 1, str(year), fontproperties=font)
    
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

    #ax1.set_title('Inequality of al cohorts over '+str(max_years)+' years')
    if len(years)<10:
        ax1.legend(years)  
    fig1.savefig("fig/gini_"+criterion+".png", facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')

    ax2.set_ylabel("Mean "+criteria_display, fontweight='bold')
    ax2.set_xlabel('Career Age', fontweight='bold')
   
    #ax2.set_title('Mean/Std of al cohorts over '+str(max_years)+' years')
    if len(years)<10:
        ax2.legend(years)  

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
    print(data.head())
    
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
        if year > (2015 - max_years):
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
                gini = calculate.gini(df_values[criterion].astype("float").values)
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