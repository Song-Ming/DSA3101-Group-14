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

print(df.head())
print(df.describe())
print(df.dtypes)

# Pre-processing
## Convert binary categorical variables to numeric
binary_features = ['housing', 'loan', 'subscribed']
for col in binary_features:
    df[col] = df[col].map({'no': 0, 'yes': 1})

## Convert poutcome to ordinal values
poutcome_mapping = {'failure': 0, 'nonexistent': 1, 'success': 2}
df['poutcome_encoded'] = df['poutcome'].map(poutcome_mapping)

## Convert contact type to binary
df['contact'] = df['contact'].map({'telephone': 0, 'cellular': 1})

## Bin pdays and previous
df['pdays_bins'] = df['pdays'].apply(lambda x: 'Never Contacted' if x == 999 else ('1-5' if x <= 5 else '6+'))
df['previous_bins'] = pd.cut(df['previous'], bins=[-1, 0, 1, float('inf')], labels=['Never', '1', '2+'])

## Drop unnecessary columns
df = df.drop(columns=['default', 'pdays', 'previous', 'poutcome'])

# Calculate ROI
## Define constants (arbitrary values)
cps_cell = 0.05   # Cost per second for a cellular call
cps_tele = 0.03  # Cost per second for a telephone call
sub_rev = 300    # Revenue for a successful subscription

## Calculate individual customer ROI
def calculate_customer_roi(row):
    # Determine call cost based on call duration and contact type
    if row['campaign'] == 0 or row['duration'] == 0:
        return 0
    if row['contact'] == 1:  # Cellular call
        call_cost = row['duration'] * cps_cell
    else:  # Telephone call
        call_cost = row['duration'] * cps_tele
    
    # Calculate total cost for the customer
    total_cost = call_cost * row['campaign']  # Multiply by the number of calls made to the customer
    
    # Add revenue if customer subscribed
    if row['subscribed'] == 1:
        total_rev = sub_rev
    else:
        total_rev = 0
    
    # Calculate ROI
    if total_cost == 0:
        return 1e-6  # Avoid division by zero if no cost
    roi = ((total_rev - total_cost) / total_cost) * 100
    
    return roi

## Apply the functions to each customer
df['roi'] = df.apply(calculate_customer_roi, axis=1)

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