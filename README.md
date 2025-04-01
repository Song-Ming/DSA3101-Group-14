# Bank Marketing Data Analysis

## Project Overview
The objective is to develop an AI-driven system to create personalised marketing campaigns for a bank by:
1.	Segmenting customers based on their behaviours and preferences.
2.	Predicting future customers needs to improve marketing effectiveness.
3.	Optimising campaigns in real-time to increase engagement and ROI.

**Dataset Source:**  
[Bank Marketing Dataset on Kaggle](https://www.kaggle.com/datasets/berkayalan/bank-marketing-data-set/data)

## Data
Run get_data.py to download the dataset from kaggle.

## src

**ppscore:**
Correlation analysis of variables in the dataset with customer engagement (subscribed).

**impact analysis:**
Computation of conversion rate and retention rate from dataset.

**roi prediction:**
Computation and prediction of ROI using CLV and Customer Acquisition Costs.
- CLV is determined by several customer demographics, such as education level, job type, marital and default status.
- Acquisition costs are determined by the mode of contact, the number of times the customer is contacted and the duration of the calls.

**customer segmentation:**
1. Identification of distinct customer groups such as high-value clients, occasional users and budget-conscious customers.
2. Analyze customer behaviors across different segments and leverage insights to recommend targeted marketing approaches.
3. Explore the possibility of implementing a real-time segmentation
model.

**churn model training:**
Implementation and optimization of a logistic regression model to predict customer churn.
- A customer is classified as churned if poutcome = failure & subscribed = no.

## Python Libraries and Versions
The project was built using the following Python libraries:

| Library      | Version |
| -------------|:-------:|
| numpy        | 1.26.4  |
| pandas       | 1.5.3   |
| ppscore      | 1.3.0   |
| sklearn      | 1.4.2   |


