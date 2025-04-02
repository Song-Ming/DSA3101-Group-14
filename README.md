# Bank Marketing Data Analysis

## Project Overview
The objective is to develop an AI-driven system to create personalised marketing campaigns for a bank by:
1.	Segmenting customers based on their behaviours and preferences.
2.	Predicting future customers needs to improve marketing effectiveness.
3.	Optimising campaigns in real-time to increase engagement and ROI.

**Dataset Source:**  
[Bank Marketing Dataset on Kaggle](https://www.kaggle.com/datasets/berkayalan/bank-marketing-data-set/data)

## Extract dataset
Run get_data.py to download the dataset from Kaggle.

Set the working directory to the project folder.

## src

**churn model training:**
Implementation and optimization of a logistic regression model to predict customer churn.
- A customer is classified as churned if poutcome = failure & subscribed = no.

**customer segmentation:**
Identification of distinct customer groups such as high-value clients, occasional users and budget-conscious customers.
- Analyze customer behaviors across different segments and leverage insights to recommend targeted marketing approaches.
- Explore the possibility of implementing a real-time segmentation
model.

**impact analysis:**
Computation of conversion rate and retention rate from dataset.

**optimise campaign:** 
Implementation of algorithm to dynamically campaign parameters according to month, day of week and contact mode (telephone/cellular) based on past campaign performance. 
- Applied clustered data according to key economic indicators ("emp.var.rate", "euribor3m", etc) to uncover patterns in financial behavior. 
- Helps determine whether housing loans, personal loans, credit customers, and term deposit subscriptions were more prevalent within each clusters.

**ppscore:**
Correlation analysis of variables in the dataset with customer engagement (subscribed) using predictive power score.

**preference prediction:**
Implementation of a random forest model to predict whether the customer will take a housing or personal loan.
- Predict customers' preference through customer demographics and campaign data.
- Use SMOTE to handle imbalanced data and use grid search for hyperparameter tuning.

**roi prediction:**
Computation and prediction of ROI using CLV and Customer Acquisition Costs.
- CLV is determined by several customer demographics, such as education level, job type, marital and default status.
- Acquisition costs are determined by the mode of contact, the number of times the customer is contacted and the duration of the calls.

## Python Libraries and Versions
The project was built using the following Python libraries:

| Library      | Version |
| -------------|:-------:|
| numpy        | 1.26.4  |
| pandas       | 1.5.3   |
| ppscore      | 1.3.0   |
| sklearn      | 1.4.2   |


