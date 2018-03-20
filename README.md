# cumulative-advantage

Steps to run the project

1) Copy the dataset from the following link and paste it in parallel to source files. It should contain a folder with name 'data' and all files inside it
https://drive.google.com/drive/folders/0B2Y4j6Wlz_LhMWlkLUtNMHNURWM
https://datorium.gesis.org/xmlui/submit?workspaceID=1635
2) Run the python notebook 'author-statistics.ipynb', which will create additional files inside 'data' folder
3) Run 'cohort-analysis-publications.ipynb', 'cohort-analysis-citations.ipynb' to see the analysis on publication and citation data respectively
4) Run 'cumadv-vs-sacspark-data-extraction.ipynb' to extract the data which will be used for hypothesis testing
   Run 'cumadv-vs-sacspark-data-test.ipynb' checks the authenticity of data extraction by checking against alternate method
   Run 'cumadv-vs-sacspark-analysis.ipynb' to understand cumulative advantage hypothesis is better than sacred spark

'author-statistics.ipynb'

'cohort-analysis-publications.ipynb'

'cohort-analysis-citations.ipynb'

'cumadv-vs-sacspark-data-extraction.ipynb'

The data has been tested for randomly picked few hundreds of authors against another implementation which was loop based . 
I also did test manually when there is a difference in results between two implementations. The following file contains the code snippet. I feel confident about the data but please let me know if there are other ways to test it. I feel another pair of eyes would be great. 

'cumadv-vs-sacspark-data-test.ipynb'

'cumadv-vs-sacspark-data-analysis.ipynb'



By default, the analysis on publication and citation data will happen for all authors which is done by using 'authors-scientific-start-end-year-publish-count.csv' file in 'data' folder. 
Restrictions can be applied (consider author who have at least 10/20 year career span and 10/20 papers published) by using the following files in the respective notebooks in 3 and 4th step
'authors-scientific-atleast-10-year-10-papers.csv'
'authors-scientific-atleast-20-year-20-papers.csv'



 