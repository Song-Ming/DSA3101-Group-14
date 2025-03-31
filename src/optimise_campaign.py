import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Preprocessing: remove non existent outcome, convert categorical variables
raw_df = pd.read_csv('./data/raw_data/bank_marketing_dataset.csv')
df = raw_df.copy()
df["month"] = df["month"].astype("category")
df["contact"] = df["contact"].astype("category")
df["day_of_week"] = df["day_of_week"].astype("category")
df = df[df["poutcome"] != "nonexistent"]
df['poutcome'] = df['poutcome'].map({'success': 1, 'failure': 0})

previous_df = df[["month", "day_of_week", "contact", "poutcome"]]

class Campaign:
    def __init__(self, day_of_week = None, month = None, contact = None):
        self.day_of_week = day_of_week
        self.month = month
        self.contact = contact 

    def optimize_campaign(self, df):
        success_by_day = df.groupby("day_of_week",  observed=False)["poutcome"].mean().sort_values(ascending=False)
        success_by_contact = df.groupby("contact",  observed=False)["poutcome"].mean().sort_values(ascending=False)
        success_by_month = df.groupby("month", observed=False)["poutcome"].mean().sort_values(ascending=False)

        self.day_of_week = success_by_day.idxmax()
        self.contact = success_by_contact.idxmax()
        self.month = success_by_month.idxmax()

        print(f"Optimized Campaign: Day = {self.day_of_week}, Contact = {self.contact}, Month = {self.month}")

# Initialize Campaign
campaign = Campaign()

### Update with new data accordingly ###

# Optimize campaign based on past data 
campaign.optimize_campaign(previous_df)

econ_df = raw_df.copy()

class EconomicClustering:
    def __init__(self, df, n_clusters=3):
        self.df = df.copy()
        self.clustered_df = None
        self.n_clusters = n_clusters
        self.random_state = 42
        self.kmeans = None
        self.scaler = StandardScaler() #Fix scaler 

    def perform_clustering(self):
        economic_factors = self.df[["emp.var.rate", "euribor3m", "nr.employed", "cons.price.idx", "cons.conf.idx"]]
        economic_factors_scaled = self.scaler.fit_transform(economic_factors)
        # KMeans clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.df["economic_cluster"] = self.kmeans.fit_predict(economic_factors_scaled)
        self.clustered_df = self.df
        
        silhouette_avg = silhouette_score(economic_factors_scaled, self.df["economic_cluster"])
        print(f"Silhouette Score: {silhouette_avg}")

    def get_highest_percentage(self, column):
        cleaned_col = self.df[self.df[column] != "unknown"].copy()
        cleaned_col[column] = cleaned_col[column].map({'yes':1, 'no':0})
        return cleaned_col.groupby("economic_cluster")[column].mean()

    def predict_cluster(self, emp_var_rate, euribor3m, nr_employed, cons_price_idx, cons_conf_idx):
        """Predicts the economic cluster for a given set of economic factor values."""
        # Format input as a 2D array
        input_data = np.array([[emp_var_rate, euribor3m, nr_employed, cons_price_idx, cons_conf_idx]])
        feature_columns = ["emp.var.rate", "euribor3m", "nr.employed", "cons.price.idx", "cons.conf.idx"]
        new_data_df = pd.DataFrame(input_data, columns=feature_columns)
        input_scaled = self.scaler.transform(new_data_df)
        cluster = self.kmeans.predict(input_scaled)[0] #Returns single value instead of array 
        return cluster

# Perform clustering
clustering = EconomicClustering(raw_df)
clustering.perform_clustering()

binary_columns = ["housing", "loan", "default", "subscribed"]
for column in binary_columns:
    cluster_performance = clustering.get_highest_percentage(column)  # Unpack directly
    print(f"{column.capitalize()} Performance by Cluster:\n{cluster_performance}\n")
    

# Test on random index 
index = 19
df_reset = clustering.clustered_df.reset_index(drop=True)
print(df_reset.loc[index, ["emp.var.rate", "euribor3m", "nr.employed", "cons.price.idx", "cons.conf.idx", 'economic_cluster']])

selected_row = df_reset.loc[19, ["emp.var.rate", "euribor3m", "nr.employed", "cons.price.idx", "cons.conf.idx"]]
row_array = selected_row.to_numpy()
predicted_cluster = clustering.predict_cluster(*row_array)
print(f"Predicted Cluster: {predicted_cluster}")

