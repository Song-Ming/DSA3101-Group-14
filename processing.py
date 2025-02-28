# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime
import math
import sklearn
import kagglehub

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(100)

# Loading data
path = kagglehub.dataset_download("berkayalan/bank-marketing-data-set")
data = pd.read_csv(f"{path}/bank_marketing_dataset.csv")
print(data.head())
print(data.describe())
print(data.dtypes)

# Check for null
print(data.isnull().sum())

# There are entries where days since last contact is 999 but number of previous contacts is not 0
temp = data.loc[data['pdays'] == 999]
temp.loc[temp['previous'] != 0]

# ## Processing

# Drop default since almost all values are 'no' or 'unknown' and duration since it is not known beforehand
data = data.drop(['default','duration'],axis = 1)

# Bin variables
binned_age = []
age = data['age']
for i in range(data.shape[0]):
    if age[i] <= 35:
        binned_age.append('35 or less')
    elif age[i] <= 40:
        binned_age.append('36 to 40')
    elif age[i] <= 50:
        binned_age.append('41 to 50')
    elif age[i] <= 60:
        binned_age.append('51 to 60')
    else:
        binned_age.append('61 or more')
binned_age = pd.Series(binned_age)
binned_age.value_counts()

binned_campaign = []
campaign = data['campaign']
for i in range(data.shape[0]):
    if campaign[i] == 1:
        binned_campaign.append('1')
    elif campaign[i] <= 5:
        binned_campaign.append('2 to 5')
    else:
        binned_campaign.append('6 or more')
binned_campaign = pd.Series(binned_campaign)
binned_campaign.value_counts()

binned_pdays = []
pdays = data['pdays']
for i in range(data.shape[0]):
    if pdays[i] <= 5:
        binned_pdays.append('1 to 5')
    elif pdays[i] == 999:
        binned_pdays.append('Never contacted')
    else:
        binned_pdays.append('6 or more')
binned_pdays = pd.Series(binned_pdays)
binned_pdays.value_counts()

binned_previous = []
previous = data['previous']
for i in range(data.shape[0]):
    if previous[i] == 0:
        binned_previous.append('Never contacted')
    elif previous[i] == 1:
        binned_previous.append('1')
    else:
        binned_previous.append('2 or more')
binned_previous = pd.Series(binned_previous)
binned_previous.value_counts()

# Replace variables with binned versions
data['age'] = binned_age
data['campaign'] = binned_campaign
data['pdays'] = binned_pdays
data['previous'] = binned_previous

# Remove social economic variables and the label
seg_data = data.drop(['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','subscribed'],axis = 1)
seg_data.dtypes


