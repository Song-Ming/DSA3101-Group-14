import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../data/raw_data/bank_marketing_dataset.csv')

# Pre-processing
## Convert binary categorical variables to numeric
binary_features = ['housing', 'loan', 'subscribed']
for col in binary_features:
    df[col] = df[col].map({'no': 0, 'yes': 1})

## Convert poutcome to ordinal values
poutcome_mapping = {'failure': 0, 'nonexistent': 1, 'success': 2}
df['poutcome'] = df['poutcome'].map(poutcome_mapping)

## Convert default to ordinal values
default_mapping = {'no': 0, 'unknown': 1, 'yes': 2}
df['default'] = df['default'].map(default_mapping)

## Convert contact type to binary
df['contact'] = df['contact'].map({'telephone': 0, 'cellular': 1})

## Bin pdays and previous
df['pdays_bins'] = df['pdays'].apply(lambda x: 'Never Contacted' if x == 999 else ('1-5' if x <= 5 else '6+'))
df['previous_bins'] = pd.cut(df['previous'], bins=[-1, 0, 1, float('inf')], labels=['Never', '1', '2+'])

## Drop unnecessary columns
df = df.drop(columns=['pdays', 'previous'])

# Calculate ROI
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
        return 0.01  # minimum acquisition cost to avoid div by 0
    cps = 0.05 if row['contact'] == 1 else 0.03
    return row['campaign'] * row['duration'] * cps

df['acquisition_cost'] = df.apply(calculate_acquisition_cost, axis=1)

# Calculate ROI
df['roi'] = ((df['clv'] - df['acquisition_cost']) / df['acquisition_cost']) * 100

# Encode categorical variables
categorical_features = ['job', 'marital', 'education', 'month', 'day_of_week', 'pdays_bins', 'previous_bins']
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = encoder.fit_transform(df[categorical_features])
categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out())

## Drop original columns and merge encoded ones
df = df.drop(columns=categorical_features)
df = pd.concat([df, categorical_df], axis=1)

# Set style
sns.set_theme(style="whitegrid")

# Cost vs ROI Plot
plot_df = df[df['roi'] < 10000]

## Find turning point for ROI
roi_smoothed = plot_df[['acquisition_cost', 'roi']].sort_values(by='acquisition_cost')
rolling_roi = roi_smoothed['roi'].rolling(100).mean()
turning_point_roi_idx = rolling_roi.diff().idxmax()  # max increase in ROI
turning_point_roi_cost = plot_df.loc[turning_point_roi_idx, 'acquisition_cost']
print(f"Turning point (Cost vs ROI): Acquisition cost = {turning_point_roi_cost:.2f}")

## Plot Cost vs ROI with turning point
plt.figure(figsize=(10,6))
sns.lineplot(x=roi_smoothed['acquisition_cost'], y=rolling_roi, color='green', label='Rolling ROI')
plt.axvline(x=turning_point_roi_cost, color='red', linestyle='--', label='Turning Point')
plt.xlabel("Acquisition Cost")
plt.ylabel("Rolling ROI (%)")
plt.title("Turning Point: Cost vs ROI")
plt.legend()
plt.show()


# Cost vs Revenue (CLV) Plot
## Find turning point for Revenue
rev_smoothed = df[['acquisition_cost', 'clv']].sort_values(by='acquisition_cost')
rolling_rev = rev_smoothed['clv'].rolling(100).mean()
turning_point_rev_idx = rolling_rev.diff().idxmax()  # max increase in revenue
turning_point_rev_cost = df.loc[turning_point_rev_idx, 'acquisition_cost']
print(f"Turning point (Cost vs Revenue): Acquisition cost = {turning_point_rev_cost:.2f}")

## Plot Cost vs Revenue with turning point
plt.figure(figsize=(10,6))
sns.lineplot(x=rev_smoothed['acquisition_cost'], y=rolling_rev, color='blue', label='Rolling Revenue')
plt.axvline(x=turning_point_rev_cost, color='red', linestyle='--', label='Turning Point')
plt.xlabel("Acquisition Cost")
plt.ylabel("Rolling CLV (Revenue)")
plt.title("Turning Point: Cost vs Revenue")
plt.legend()
plt.show()