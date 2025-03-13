# ppscore library information:
# For pandas>2.0.0, scikit-learn>2.0.0
# You need to set up a venv to import ppscore

import numpy as np
import pandas as pd
import kagglehub
import ppscore
import matplotlib.pyplot as plt
import seaborn as sns

# Load data without any modifications
path = kagglehub.dataset_download("berkayalan/bank-marketing-data-set")
df = pd.read_csv(f"{path}/bank_marketing_dataset.csv")

print("Dataset loaded successfully.")

# Only include those who have subscribed previously
new_df = df[df['poutcome'] == 'success']
# Compute PPS (note: unknown is treated as a third value)
pps_subscribed = ppscore.predictors(new_df,'subscribed',random_seed = 100, invalid_score = 1000)

# Pivot the data for heatmap visualization
heatmap_data = pps_subscribed.pivot(index="y", columns="x", values="ppscore")

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True, fmt=".2f")
plt.title("PPS for Predicting Subscription (subscribed)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


