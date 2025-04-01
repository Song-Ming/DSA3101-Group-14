import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import random
import numpy as np

# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

# Load data
df = pd.read_csv('../data/raw_data/bank_marketing_dataset.csv')

#### Use ROI code from Q8 ######
# Pre-processing
## Convert binary categorical variables to numeric
binary_features = ['housing', 'loan', 'subscribed']
for col in binary_features:
    df[col] = df[col].map({'no': 0, 'yes': 1})

## Convert contact type to binary
df['contact'] = df['contact'].map({'telephone': 0, 'cellular': 1})

## Convert to ordinal values
poutcome_mapping = {'failure': 0, 'nonexistent': 1, 'success': 2}
df['poutcome'] = df['poutcome'].map(poutcome_mapping)

default_mapping = {'no': 0, 'unknown': 1, 'yes': 2}
df['default'] = df['default'].map(default_mapping)

## Bin pdays and previous
df['pdays_bins'] = df['pdays'].apply(lambda x: 'Never Contacted' if x == 999 else ('1-5' if x <= 5 else '6+'))
df['previous_bins'] = pd.cut(df['previous'], bins=[-1, 0, 1, float('inf')], labels=['Never', '1', '2+'])

## Drop unnecessary columns
df = df.drop(columns=['pdays', 'previous', 'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'])

# Calculate ROI
## (Take conversion rate/previous campaign success rate into account)
## Calculate conversion rate among those who did not subscribe in previous campaign
success_df = df[df['poutcome'] != 2]
conversion_rate = sum(success_df['subscribed'] == 1) / success_df.shape[0]

## Calculate previous campaign success rate
prev_campaign = df['poutcome'].value_counts()
prev_success = prev_campaign[2] / (prev_campaign[0] + prev_campaign[2])

## Function to calculate CLV
def calculate_clv(row):
    clv = 100  # base CLV
    # Factor in previous campaign outcome
    if row['poutcome'] == 2:
        prob_conversion = 1 + prev_success
    elif row['poutcome'] == 0:
        prob_conversion = conversion_rate
    else:
        prob_conversion = 1
    clv *= prob_conversion # Adjust CLV based on conversion probability
    # Factor in other customer attributes
    if row['subscribed'] == 1:
        clv += 200
    if row['housing'] == 1:
        clv += 100  # Housing loan customers indicate a long-term banking relationship
    if row['loan'] == 1:
        clv += 100  # Personal loan customers may need more banking products
    if row['marital'] == 'single':
        clv += 50 # More financial independence
    if row['education'] in ['university.degree', 'professional.course']:
        clv += 50  # More financially aware customers may invest more
    if row['job'] in ['management', 'entrepreneur', 'self-employed']:
        clv += 50  # Higher earning potential, more banking needs
    if row['default'] == 2:
        clv -= 200 # Penalty for defaulting customers
    return max(clv, 0)  # CLV should be non-negative

df['clv'] = df.apply(calculate_clv, axis=1)

## Function to calculate customer acquisition cost
def calculate_acquisition_cost(row):
    if row['campaign'] == 0 or row['duration'] == 0:
        return 0.01  # minimum acquisition cost to avoid division by 0
    cps = 0.05 if row['contact'] == 1 else 0.03 # cost per second
    return row['campaign'] * row['duration'] * cps

df['acquisition_cost'] = df.apply(calculate_acquisition_cost, axis=1)

# Calculate ROI
df['roi'] = ((df['clv'] - df['acquisition_cost']) / df['acquisition_cost']) * 100

df_unscaledcopy = df.copy()

# Encode categorical variables
categorical_features = ['job', 'marital', 'education', 'month', 'day_of_week', 'pdays_bins', 'previous_bins']
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = encoder.fit_transform(df[categorical_features])
categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out())

## Drop original columns and merge encoded ones
df = df.drop(columns=categorical_features)
df = pd.concat([df, categorical_df], axis=1)

## Fill missing values
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

#### Use Clustering code from Q1 ######
wcss = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# Apply KMeans clustering with k=3
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df)
df_unscaledcopy['cluster'] = df['cluster'] ##for visualisation of clusters

# Perform PCA to reduce the data to 2 components for visualisation
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df.drop(columns=['cluster']))

# Analyse clusters
cluster_summary = df_unscaledcopy.groupby('cluster').agg({
    'roi': ['mean', 'median'],
    'clv': ['mean', 'median'],
    'acquisition_cost': ['mean', 'median'],
    'subscribed': 'mean',
}).round(2)

print(cluster_summary)

categorical_cols = ['job', 'education', 'marital']

for col in categorical_cols:
    print(f"\n{col.upper()} DISTRIBUTION BY CLUSTER")
    print(df_unscaledcopy.groupby('cluster')[col].value_counts(normalize=True).unstack().round(2))

roi_quartiles = df_unscaledcopy['roi'].quantile([0.25, 0.5, 0.75])
print(roi_quartiles)

# Define the ROI segments
roi_quartiles = df['roi'].quantile([0.25, 0.5, 0.75])

# Create a new column for ROI segments based on quartiles
def classify_roi(row):
    if row['roi'] < roi_quartiles[0.25]:
        return 'Low ROI'
    elif row['roi'] < roi_quartiles[0.75]:
        return 'Medium ROI'
    else:
        return 'High ROI'

df['roi_segment'] = df.apply(classify_roi, axis=1)

# Select features and target variable
X = df.drop(columns=['roi', 'roi_segment', 'cluster'])  # Drop ROI, roi_segment, and cluster columns
y = df['roi_segment']  # Target variable is the ROI segment

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Low ROI', 'Medium ROI', 'High ROI'], rounded=True)
plt.show()