import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

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

## Remove unnecessary columns
df = df.drop(['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'],axis = 1)

## Fill missing values
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split and scale data
## Define features and target
X = df.drop(columns=['roi'])
y = df['roi']

## Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model using Random Forest
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

## Predict and evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

## Feature importance
feature_importances = pd.Series(model.feature_importances_, index=df.drop(columns=['roi']).columns)
print("Feature Importances:")
print(feature_importances.sort_values(ascending=False))