# ATUS-Classification

Predicting People’s Actions Using American Time Use Data

Executive Summary
The American Time Use Survey is a dataset monitoring time use in the US. The data measures how people of different demographics spent time throughout the day. ATUS data is very popular among economists and is historically used to determine labor inequality among people of different socioeconomic statuses and gender. However, very little research using American Time Use Data has been done on a predictive scale. Using the following models, I used 624 feature variables to predict the 18 activity categories.  

1.	Bernoulli 
2.	Gaussian 
3.	Nearest Centroid (Cross Validated) 
4.	Logistic Regression 
5.	Ridge Regression (Cross Validated)(Scaled)
6.	Decision Tree (Cross Validated)
7.	KNN (Cross Validated)
8.	Neural Network (Cross Validated)(Scaled)
9.	SVC 

Decision Tree provided the best predictive power with a 0.60 accuracy score. The variables that were the most influential in prediction are time of day, location, and age of household members. Logistic Regression would have been a second-best model with a 0.40 accuracy score. However, this model was too computationally expensive for large datasets. Future work extending this project would be to expand the database to include the 15-year dataset extending from 2013 – 2018 and to do a separate analysis using the 4-code and 6-code activity lexicons. Although given the subtle differences in the higher specificity categories, it would be harder for the model to predict accurately. 

Introduction
The American Time Use Survey is a dataset published by the Department of Labor Statistics and conducted by the US Census Bureau monitoring time use in the US. The data measures how people of different demographics spent time throughout the day. Individuals are randomly selected from a subset of households that have completed their interviews for the Current Population Survey (CPS). Respondents are interviewed one time about how they spent their time on the previous day, where they were, and whom they were with. 

There are 8 datasets total and four that were used are: 

1.	Respondent file: information about ATUS respondents and their labor force and earnings. 
2.	Roster file: Information about household members and their children (under 18) as well as information such as age and sex.
3.	Activity file: information about how ATUS respondents spent their diary day. This includes activity codes, activity start and stop times, and locations
4.	Who file: codes that indicate who was present during each activity
