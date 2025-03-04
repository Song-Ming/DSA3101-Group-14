import numpy as np
import pandas as pd
import ppscore as pps
import matplotlib.pyplot as plt
import seaborn as sns

# Load df
df = pd.read_csv('processed_data/cor_processing.csv')

# Compute PPS matrix
pps_matrix = pps.matrix(df)

# Round PPS values to 2 decimal places
pps_matrix['ppscore'] = pps_matrix['ppscore'].round(2)

# Remove self-predicting rows (x == y)
pps_matrix = pps_matrix[pps_matrix["x"] != pps_matrix["y"]]

# Filter out weak predictors (only keep ppscore > 0.1)
pps_matrix_filtered = pps_matrix[pps_matrix["ppscore"] > 0.1]

# Save filtered PPS matrix to CSV
pps_matrix_filtered.to_csv('processed_data/ppscore_matrix_filtered.csv', index=False)

# Visualization: Heatmap of filtered PPS Matrix
plt.figure(figsize=(12, 8))
matrix_df = pps_matrix_filtered.pivot(columns='x', index='y', values='ppscore')

sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True, fmt=".2f")
plt.title("Filtered Predictive Power Score (PPS) Matrix (ppscore > 0.1)")
plt.show()

print("Filtered PPS matrix has been saved to 'processed_data/ppscore_matrix_filtered.csv'")
