import numpy as np
import pandas as pd
import kagglehub

# Load data without any modifications
path = kagglehub.dataset_download("berkayalan/bank-marketing-data-set")
df = pd.read_csv(f"{path}/bank_marketing_dataset.csv")

print("Dataset loaded successfully.")

# Filter customers who subscribed in the previous campaign
previous_success_df = df[df['poutcome'] == 'success']
retention_rate = (previous_success_df['subscribed'] == 'yes').sum() / previous_success_df.shape[0] * 100

print('Retention rate is {}%'.format(retention_rate))

#Retention rate is 65.1128914785142%