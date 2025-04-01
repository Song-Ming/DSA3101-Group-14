from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
 
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns

def preprocess_data():
    df = pd.read_csv('./data/raw_data/bank_marketing_dataset.csv')
    # No missing values

    # Standardize naming
    df.columns = df.columns.str.replace('.', '_')
    df['education'] = df['education'].str.replace('.', '_')

    # Defining churn: Consecutive failures to subscribe (poutcome == failure & subscribed == no)
    df['churn'] = np.where((df['poutcome'] == 'failure') & (df['subscribed'] == 'no'), 1, 0)
    df.drop(['poutcome','subscribed'], axis=1, inplace=True)

    # Categorical variables
    categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'day_of_week']
    
    # One hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    ohe = encoder.fit_transform(df[categorical])
    df_ohe = pd.DataFrame(ohe, columns=encoder.get_feature_names_out(categorical))
    df_encoded = pd.concat([df,df_ohe], axis=1).drop(categorical, axis=1)

    # Split features
    ## Dropped euribor3m and nr_employed. cor(euribor3m, nr_employed) = 0.95. cor(euribor3m, emp_var_rate) = 0.97
    X, y = df_encoded.drop(['euribor3m', 'nr_employed','churn'], axis=1), df_encoded['churn']

    return X,y,encoder

def train_model():
    # Get split data
    X,y,encoder = preprocess_data()

    # Train test split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create base Logistic regression model and validate using KFold
    logreg = LogisticRegression(max_iter=1000)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    ## Validate model
    scores = cross_val_score(logreg, X_train_scaled, y_train.to_numpy().ravel(), scoring='f1', cv=cv)
    ## Evaluate
    print('F1 Score of base Logistic Regression model: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    
    # Tune model
    ## Grid Search CV
    params = {'C': [0.01, 0.1, 1, 10],
          'penalty': ['l1', 'l2'],
          'solver': ['liblinear', 'saga', 'lbfgs', 'sag'] # include solvers like 'liblinear' or 'saga' that are compatible with both penalty types
        }
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5, refit='f1') # For imbalanced data, accuracy would be misleading
    grid_search.fit(X_train_scaled, y_train.to_numpy().ravel())
    best_model = grid_search.best_estimator_

    ## Evalute tuned model on test set
    y_pred = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    # Plot prediction Confusion Matrix
    prediction_cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(prediction_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix - Logistic Regression (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC AUC
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    ## Calculate the ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.2f}")
    ## Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    ## Plot ROC AUC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Curve - Logistic Regression")
    plt.legend(loc="lower right")
    plt.show()

    # Feature importance plot (for Logistic Regression, we'll use the absolute values of coefficients)
    importances = np.abs(best_model.coef_[0])
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12,8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [X.columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_model()