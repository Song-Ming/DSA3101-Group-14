# ppscore_venv\Scripts\activate


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

# Compute PPS matrix
pps_matrix = ppscore.matrix(df)

# # Remove self-predicting values (x == y)
# pps_matrix = pps_matrix[pps_matrix["x"] != pps_matrix["y"]]

# # Filter out weak predictors (PPS > 0.1 for readability)
# pps_matrix_filtered = pps_matrix[pps_matrix["ppscore"] > 0.1]

# Pivot the data for heatmap visualization
pps_subscribed = pps_matrix[pps_matrix["y"] == "subscribed"]
heatmap_data = pps_subscribed.pivot(index="y", columns="x", values="ppscore")

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True, fmt=".2f")
plt.title("PPS for Predicting Subscription (subscribed)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()
