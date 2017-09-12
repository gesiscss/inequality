# cumulative-advantage

Steps to run the project

1) Copy the dataset from the following link and paste it in parallel to source files. It should contain a folder with name 'data' and all files inside it
https://drive.google.com/drive/folders/0B2Y4j6Wlz_LhMWlkLUtNMHNURWM

2) Run the python notebook 'cumulative-adv-author-statistics-for-analysis.ipynb', which will create additional files inside 'data' folder

3) Run 'cumulative-adv-cohort-analysis-for-1970-2000-publications.ipynb' to see the analysis on publication data

4) Run 'cumulative-adv-cohort-analysis-for-1970-2000-citations.ipynb' to see the analysis on citation data

By default, the analysis on publication and citation data will happen for all authors which is done by using 'authors-scientific-start-end-year-publish-count.csv' file in 'data' folder. 
Restrictions can be applied (consider author who have at least 10/20 year career span and 10/20 papers published) by using the following files in the respective notebooks in 3 and 4th step
'authors-scientific-atleast-10-year-10-papers.csv'
'authors-scientific-atleast-20-year-20-papers.csv'



 