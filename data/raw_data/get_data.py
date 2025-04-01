# Extract data without any modifications
import pandas as pd
import kagglehub

path = kagglehub.dataset_download("berkayalan/bank-marketing-data-set")
df = pd.read_csv(f"{path}/bank_marketing_dataset.csv")
df.to_csv('./data/raw_data/bank_marketing_dataset.csv', index=False)