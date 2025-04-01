import numpy as np
import pandas as pd
import kagglehub

# Load data without any modifications
path = kagglehub.dataset_download("berkayalan/bank-marketing-data-set")
df = pd.read_csv(f"{path}/bank_marketing_dataset.csv")

print("Dataset loaded successfully.")

# Calculate conversion rate among those who did not subscribe in previous campaign
success_df = df[df['poutcome'] != 'success']
conversion = sum(success_df['subscribed'] == 'yes') / success_df.shape[0]
print('Conversion rate is {}%'.format(conversion*100))

#Conversion rate is 9.408514379002888%

# Filter customers who subscribed in the previous campaign
previous_success_df = df[df['poutcome'] == 'success']
retention_rate = (previous_success_df['subscribed'] == 'yes').sum() / previous_success_df.shape[0] * 100

print('Retention rate is {}%'.format(retention_rate))

#Retention rate is 65.1128914785142%

# Calculate previous campaign success rate
prev_campaign = df['poutcome'].value_counts()
prev_success = prev_campaign['success'] / (prev_campaign['failure'] + prev_campaign['success'])
print('Previous campaign success rate is {}%'.format(prev_success*100))

#Previous campaign success rate is 24.40888888888889%
