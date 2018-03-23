# Documentation

## cumulative-advantage

### Papers to read
* Cumulative advantage and Inequality in Science by Paul D. Allison, J. Scott Long, Tad K. Krauze
* link to the paper which claudia has been writing ???
* E-mails - sent by claudia

### Code base

#### Steps to run the project
* Copy the dataset from the following link and paste it in parallel to source files. It should contain a folder with name 'data' and all files inside it
https://drive.google.com/drive/folders/0B2Y4j6Wlz_LhMWlkLUtNMHNURWM
https://datorium.gesis.org/xmlui/submit?workspaceID=1635
* Run the python notebook 'author-statistics.ipynb', which will create additional files inside 'data' folder
* Run 'cohort-analysis-publications.ipynb', 'cohort-analysis-citations.ipynb' to see the analysis on publication and citation data respectively

#### File level explanation

##### 'author-statistics.ipynb'
* Different plots to understand the nature of the data - statistics
* Compute the start and end year, career span for each author and store in a separate file - this will be used further to filter the dataset to allow only credible authors
* Analysis to figure out how many years of career span is good to be considered for cohort analysis

##### 'cohort-analysis-publications.ipynb'
##### 'cohort-analysis-citations.ipynb'
* Scientists are grouped based on the year they started their career (cohorts). 
* Each cohort is analysed over 15 years on their production (how many papers that they publish) or on their recognition (how many citations that they received)
* Gini is computed for each cohort over 15 years and it is plotted. Steady increase in inequality is taken as a proof for presence of cumulative advantage

## Cumulative advantage vs Sacred Spark

### Papers to read
* Is there any?

### Code base

#### Steps to run the project
* Run 'author-statistics.ipynb' - this will create the credible authors file
* If 'author-career-short-snapshot.csv' data is not shared then run 'cumadv-vs-sacspark-data-extraction.ipynb'. This will create the respective file in 'data' folder. Run 'cumadv-vs-sacspark-data-test.ipynb' to test the dataset created. It should pass.
* If data is shared then copy it to 'data' folder
* Run 'cumadv-vs-sacspark-analysis.ipynb' to understand whether cumulative advantage hypothesis is better than sacred spark for computer scientists domain

#### File level explanation

##### 'cumadv-vs-sacspark-data-extraction.ipynb'
##### 'cumadv-vs-sacspark-data-test.ipynb'
*Two contrasting theories when it comes to how scientists succeed in scientific community 
*Sacred Spark - intellectual scientists will perform good irrespective of initial failures
*Cumulative Advantage - Initial differences among scientists in performance/recognition grows and it gets stronger and stronger and will lead to successful/mediocre/not-so-successful scientists. Initial differences performs a crucial role than the intelligence

In order to conduct the analysis, many features need to be extracted which is referring to initial and final productivity/recognition of scientists such as cumulative publication at their 1st year, 2nd and 3rd year and the cumulative citations they have obtained during subsequent 3 years and 15th year. 

When run in server, the extraction will approximately take more than 11 hours and it would be of 150 mb size. The code makes complete use of panda's parallelism which is by using the vectorized operations. 

Once when the extaction is completed, the data can be tested for randomly picked few hundreds of authors against another implementation which was loop based. 

##### 'cumadv-vs-sacspark-data-analysis.ipynb'
The ideal thing to do would be to conduct an experiment where in scientists who start their career with same motivation, intellectual levels and similiar productivity/recognition levels would go through a process where in one would get to publish more / get recognized more and measure their performance along the same after few years. But this experiment will take a lot of effort and time :-) and hence we use the observed data. We used matching method technique. The following links will give good information about experiment design followed by matching methods and why one technique (propensity score) should not be used in matching method (which can be skipped)

https://www.youtube.com/watch?v=DaBq0naj0YY
https://www.youtube.com/watch?v=UEFBGewP2ik
https://www.youtube.com/watch?v=rBv39pK1iEs&t=3s

We have used mahalanobis distance to compare the distances and performed the analysis to figure out whether average cumulative advantage phenomenon effect is statistically significant or not?
