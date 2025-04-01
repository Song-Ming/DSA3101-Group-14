import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import kagglehub


df = pd.read_csv('./data/raw_data/bank_marketing_dataset.csv')
df = df.drop(['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','duration','campaign','previous','subscribed', 'pdays', 'default'], axis=1)

df["combined_education"] = df["education"].apply(lambda x: "basic" if x in set({"basic.4y", "basic.6y", "basic.9y"}) else x)
df["job"] = df["job"].apply(lambda x: "high" if x in set({"admin.", "technician", "management", "entrepreneur", "self-employed"}) else "low")

encoder = OneHotEncoder()
one_hot_encoded = encoder.fit_transform(df[["job", "marital", "combined_education", "contact", "month", "day_of_week", "housing", "loan"]]).toarray()
encoded_personal_info = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(["job", "marital", "combined_education", "contact", "month", "day_of_week", "housing", "loan"]))
encoded_personal_info = pd.concat([df, encoded_personal_info], axis=1)
encoded_personal_info = encoded_personal_info.drop(["job", "marital", "combined_education", "contact", "month", "day_of_week", "poutcome", "housing_unknown",
                                                        "loan_unknown", "education", "loan_no", "contact_cellular", "housing_no", "job_low"], axis=1)
encoded_personal_info["housing"] = encoded_personal_info["housing"].apply(lambda x: 1 if x == 'yes' else 0)
encoded_personal_info["loan"] = encoded_personal_info["loan"].apply(lambda x: 1 if x == 'yes' else 0)

housing_df = encoded_personal_info[encoded_personal_info.housing != "unknown"]
housing_train = housing_df.drop(["housing", "loan", "housing_yes"], axis=1)
housing_label = housing_df["housing"]

X_train, X_test, y_train, y_test = train_test_split(housing_train, housing_label, test_size=0.2, random_state=1)

np.random.seed(seed=0)
dt = RandomForestClassifier(class_weight='balanced', criterion='gini', max_features='log2', min_samples_leaf=4, n_estimators=200, max_depth=10, min_samples_split=2)
dt_model = dt.fit(X_train,y_train)
housing_pred = dt_model.predict(X_test)

precision = precision_score(y_test, housing_pred)
recall = recall_score(y_test, housing_pred)
f1 = f1_score(y_test, housing_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1_score: {f1:.2f}")

feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

#print(feature_importances)
top_n = 2
top_features = feature_importances.iloc[:top_n]['Feature'].tolist()

np.random.seed(seed=0)
X_train_h = X_train[top_features]
X_test_h = X_test[top_features]
dt = RandomForestClassifier(class_weight='balanced', criterion='gini', max_features='log2', min_samples_leaf=4, n_estimators=200, max_depth=10, min_samples_split=2)
dt_model = dt.fit(X_train_h,y_train)
housing_pred = dt_model.predict(X_test_h)

precision = precision_score(y_test, housing_pred)
recall = recall_score(y_test, housing_pred)
f1 = f1_score(y_test, housing_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1_score: {f1:.2f}")

important_tree = dt_model.estimators_[0]

# plt.figure(figsize=(35, 25), dpi=180)  
# plot_tree(
#     important_tree,
#     feature_names=X_train_h.columns,
#     class_names=[str(i) for i in np.unique(y_train)],
#     filled=True,
#     rounded=True,
#     max_depth=3, 
#     fontsize=14  
# )
# plt.show()


loan_df = encoded_personal_info[encoded_personal_info.loan != "unknown"]
loan_train = loan_df.drop(["housing", "loan", "loan_yes"], axis=1)
loan_label = loan_df["loan"]

X_train, X_test, y_train, y_test = train_test_split(loan_train, loan_label, test_size=0.2, random_state=1)

np.random.seed(seed=0)
dt = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth=30, max_features='log2', min_samples_leaf=1, min_samples_split=5, n_estimators=300)
smote = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
dt_model = dt.fit(X_train_resampled, y_train_resampled)
loan_pred = dt_model.predict(X_test)
f1 = classification_report(y_test, loan_pred, output_dict=True)['weighted avg']['f1-score']
micro_f1 = f1_score(y_test, loan_pred, average='micro')

print(f"Micro F1-Score: {micro_f1:.4f}")
print(f"F1_score: {f1:.2f}")
print('Decision Tree accuracy for training set: %f' % dt_model.score(X_train, y_train))
print('Decision Tree accuracy for test set: %f' % dt_model.score(X_test, y_test))

feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

#print(feature_importances)
top_n = 2
top_features = feature_importances.iloc[:top_n]['Feature'].tolist()

np.random.seed(seed=0)
X_train_resampled_l = X_train_resampled[top_features]
X_test_l = X_test[top_features]
dt = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth=30, max_features='log2', min_samples_leaf=1, min_samples_split=5, n_estimators=300)
dt_model = dt.fit(X_train_resampled_l, y_train_resampled)
loan_pred = dt_model.predict(X_test_l)
f1 = classification_report(y_test, loan_pred, output_dict=True)['weighted avg']['f1-score']
micro_f1 = f1_score(y_test, loan_pred, average='micro')

print(f"Micro F1-Score: {micro_f1:.4f}")
print(f"F1_score: {f1:.2f}")

important_tree = dt_model.estimators_[0]

# plt.figure(figsize=(50, 35), dpi=280)  
# plot_tree(
#     important_tree,
#     feature_names=X_train_resampled_l.columns,
#     class_names=[str(i) for i in np.unique(y_train_resampled)],
#     filled=True,
#     rounded=True,
#     max_depth=3, 
#     fontsize=14  
# )
# plt.show()