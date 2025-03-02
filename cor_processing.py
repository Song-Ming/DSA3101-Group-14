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
from sklearn.preprocessing import LabelEncoder

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(100)

# Loading data
path = kagglehub.dataset_download("berkayalan/bank-marketing-data-set")
df = pd.read_csv(f"{path}/bank_marketing_dataset.csv")
print(df.head())
print(df.describe())
print(df.dtypes)

# Check for null
print(df.isnull().sum())

# +
# Note: There are entries where days since last contact is 999 but number of previous contacts is not 0
#temp = df.loc[df['pdays'] == 999]
#temp.loc[temp['previous'] != 0]
# -

# ## Processing

# Drop default since almost all values are 'no' or 'unknown' and duration since it is not known beforehand
df = df.drop(['default','duration'],axis = 1)

# +
# Encode categorical variables using one hot encoding
categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for col in categorical_cols:
    df[col] = df[col].replace('unknown', 'other')

df_encode = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
# -

# Encode label
# Convert 'subscribed' to binary (0 == No, 1 == Yes)
df['subscribed'] = df['subscribed'].map({'no': 0, 'yes': 1})

# Convert 999 in pdays to NaN (Because 999 represents not contacted)
df['pdays'] = df['pdays'].replace(999, np.nan)

df_encode.to_csv('processed_data/cor_processing.csv', index=False)
