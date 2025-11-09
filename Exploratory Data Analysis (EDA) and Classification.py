# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.svm import SVC
from datetime import datetime, timedelta


# Read data from CSV into a pandas Dataframe
Dataframe = pd.read_csv('data.csv', low_memory=False)

# Data Preprocessing

# Drop unnecessary columns
Dataframe = Dataframe.drop(columns=['Unnamed: 0',
                                    'product_age_group',
                                    'product_gender',
                                    'product_brand',
                                    'product_category1',
                                    'product_category2',
                                    'product_category3',
                                    'product_category4',
                                    'product_category5',
                                    'product_category6',
                                    'product_category7'],axis=1)

# Define a function to fill missing values in 'Sale' column based on 'SalesAmountInEuro'
def fill_sale_column(row):
    if pd.isna(row['Sale']):
        if row['SalesAmountInEuro'] == -1:
            return 0.0
        else:
            return 1.0
    else:
        return row['Sale']

# Apply the function to fill missing values in 'Sale' column
Dataframe['Sale'] = Dataframe.apply(fill_sale_column, axis=1)

# Calculate mean price based on product_id
mean_price_based_on_product_id = Dataframe[(Dataframe['product_price'] != -1) & (Dataframe['product_id'] != '-1' )][['product_id','product_price']].groupby(by='product_id').mean()

# Define a function to fill missing values in 'product_price' column based on 'product_id'
def fill_price_based_on_product_id(row):
    if row['product_price'] == -1:
        if row['product_id'] != '-1':
            try:
                return mean_price_based_on_product_id.loc[row['product_id']].item()
            except:
                return -1
        else:
            return -1
    else:
        return row['product_price']

# Apply the function to fill missing values in 'product_price' column
Dataframe['product_price'] = Dataframe.apply(fill_price_based_on_product_id, axis=1)

# Calculate unique product countries per partner_id
country_partner_uniques_dataframe = Dataframe[Dataframe['product_country'] != '-1'][['product_country','partner_id']].groupby(by='partner_id')['product_country'].unique()

# Define a function to fill missing values in 'product_country' column based on 'partner_id'
def fill_country_based_on_partner_id(row):
    if row['product_country'] == '-1':
        if row['partner_id'] != '-1':
            try:
                x = country_partner_uniques_dataframe.loc[row['partner_id']]
                if isinstance(x, np.ndarray):
                    return x[0]
                else:
                    return x
            except:
                return '-1'
        else:
            return '-1'
    else:
        return row['product_country']

# Apply the function to fill missing values in 'product_country' column
Dataframe['product_country'] = Dataframe.apply(fill_country_based_on_partner_id, axis=1)

# Calculate statistics based on product country and price
Dataframe1 = Dataframe[(Dataframe['product_country']!='-1') & (Dataframe['product_price']!=-1)][['product_country','product_price']]
ag_Dataframe1 = Dataframe1.groupby(by='product_country')

res_table = ag_Dataframe1.mean()
res_table['max'] = ag_Dataframe1.max()['product_price'].values
res_table['min'] = ag_Dataframe1.min()['product_price'].values
res_table['count'] = ag_Dataframe1.count()['product_price'].values
res_table['median'] = ag_Dataframe1.median()['product_price'].values

#Plot histograms to visualize distribution of product prices for different countries
# Dataframe[(Dataframe['product_country'] == '57A1D462A03BD076E029CF9310C11FC5') & (Dataframe['product_price']!= -1)]['product_price'].plot(kind='hist',bins=100)
# plt.show()
# Dataframe[(Dataframe['product_country'] == '2AC62132FBCFA093B9426894A4BC6278') & (Dataframe['product_price']!= -1)]['product_price'].plot(kind='hist',bins=100)
# plt.show()

# Calculate median price based on product country
median_price_based_on_country = ag_Dataframe1.median()

# Define a function to fill missing values in 'product_price' column based on 'product_country'
def fill_price_based_on_product_country(row):
    if row['product_price'] == -1:
        if row['product_country'] != '-1':
            try:
                return median_price_based_on_country.loc[row['product_country']].item()
            except:
                return -1
        else:
            return -1
    else:
        return row['product_price']

# Apply the function to fill missing values in 'product_price' column
Dataframe['product_price'] = Dataframe.apply(fill_price_based_on_product_country, axis=1)

# Calculate statistics based on audience_id and price
Dataframe1 = Dataframe[(Dataframe['audience_id']!='-1') & (Dataframe['product_price']!=-1)][['audience_id','product_price']]
ag_Dataframe1 = Dataframe1.groupby(by='audience_id')

res_table = ag_Dataframe1.mean()
res_table['max'] = ag_Dataframe1.max()['product_price'].values
res_table['min'] = ag_Dataframe1.min()['product_price'].values
res_table['count'] = ag_Dataframe1.count()['product_price'].values
res_table['median'] = ag_Dataframe1.median()['product_price'].values
print(res_table)

# Plot histograms to visualize distribution of product prices for different audience IDs
Dataframe[(Dataframe['audience_id'] == '013F1DD80B0848F555FBC82C981E9747') & (Dataframe['product_price']!= -1)]['product_price'].plot(kind='hist',bins=100)
plt.show()
Dataframe[(Dataframe['audience_id'] == 'ED37757B4EBDAC15881F1E9B29A35096') & (Dataframe['product_price']!= -1)]['product_price'].plot(kind='hist',bins=100)
plt.show()

# Calculate median price based on audience ID
median_price_based_on_audience_id = ag_Dataframe1.median()

# Define a function to fill missing values in 'product_price' column based on 'audience_id'
def fill_price_na_based_on_audience_id(row):
    if row['product_price'] == -1:
        if row['audience_id'] != '-1':
            try:
                return median_price_based_on_audience_id.loc[row['audience_id']].item()
            except:
                return -1
        else:
            return -1
    else:
        return row['product_price']

# Apply the function to fill missing values in 'product_price' column
x = Dataframe.apply(fill_price_na_based_on_audience_id, axis=1)

# Handling Time Information

# Extract last date from 'click_timestamp'
last_date = Dataframe['click_timestamp'].max()
last_date = datetime.fromtimestamp(last_date)

# Calculate start of the week date
start_of_week_date = last_date - timedelta(days=7)
start_of_week_timestamp = int(start_of_week_date.timestamp())

# Filter data for last week
last_week_Dataframe = Dataframe[Dataframe['click_timestamp'] > start_of_week_timestamp]

# Calculate click counts for last week
click_counts_last_week = last_week_Dataframe[last_week_Dataframe['product_id'] != '-1']['product_id'].value_counts()

# Define a function to fill missing values in 'nb_clicks_1week' column based on 'product_id'
def fill_na_nb_click_1week(row):
    if row['nb_clicks_1week'] == -1: 
        if row['product_id'] != '-1':
            try:
                return click_counts_last_week.loc[row['product_id']]
            except:
                return 0
        else:
            return 0
    else:
        return row['nb_clicks_1week']

# Apply the function to fill missing values in 'nb_clicks_1week' column
Dataframe['nb_clicks_1week'] = Dataframe.apply(fill_na_nb_click_1week, axis=1)
# %%
# Clean the dataset

# Drop rows with missing or irrelevant values
Dataframe = Dataframe[Dataframe['product_price'] != -1]
Dataframe = Dataframe.drop(['SalesAmountInEuro','time_delay_for_conversion','product_title','user_id'], axis=1)
Dataframe = Dataframe[Dataframe['device_type'] != '-1']
Dataframe = Dataframe.drop(['audience_id'], axis=1)

# Save cleaned dataset to CSV
Dataframe.to_csv('clean-dataset.csv')
# %%
# Drop rows with missing 'product_id'
Dataframe = Dataframe[Dataframe['product_id'] != '-1']

# Split dataset into features (X) and target (y)
y = Dataframe['Sale']
X = Dataframe[Dataframe.columns[1:]]

# Undersample the majority class to handle class imbalance
undersampler = RandomUnderSampler(random_state=42)
X, y = undersampler.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Define preprocessing steps for numerical and categorical features
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocess features using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ['click_timestamp','nb_clicks_1week', 'product_price']),
        ('cat', categorical_transformer, ['device_type', 'product_country', 'product_id', 'partner_id'])
    ])

# Define a Decision Tree Classifier pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', DecisionTreeClassifier(random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Model Evaluation

# Calculate and print accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nDecision Tree  Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and plot ROC curve
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# %%
# Define preprocessing steps for numerical and categorical features for Logistic Regression
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define a Logistic Regression pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Model Evaluation for Logistic Regression

# Calculate and print accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate AUC and plot ROC curve
y_probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# %%
# Define a Support Vector Classifier (SVC) pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', SVC(kernel='linear', probability=True, random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Model Evaluation for SVC

# Calculate and print accuracy
print("SVC Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report
print("\nSVC Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nSVC Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate AUC and plot ROC curve
y_probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVC Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# %%
# Define preprocessing steps for numerical and categorical features for XGBoost
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define a XGBoost Classifier pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', xgb.XGBClassifier(random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Model Evaluation for XGBoost
# Calculate and print accuracy
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate AUC and F1 Score, and plot ROC curve
y_probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_probs) 
f1 = f1_score(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# %%
# Define preprocessing steps for numerical and categorical features
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocess features using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ['click_timestamp','nb_clicks_1week', 'product_price']),
        ('cat', categorical_transformer, ['device_type', 'product_country', 'product_id', 'partner_id'])
    ])
# %%
# Define a Random Forest Classifier pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Model Evaluation

# Calculate and print accuracy
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and plot ROC curve
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# %%

# Accuracy
print("K-Nearest Neighbors Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nK-Nearest Neighbors Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nK-Nearest Neighbors Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC Curve and AUC
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K-Nearest Neighbors Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %%
