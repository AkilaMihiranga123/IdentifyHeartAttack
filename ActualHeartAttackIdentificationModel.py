# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and save the data into variable
df = pd.read_csv('heart.csv')

# Print the data
df.head()

# Get data types
df.dtypes

# Get the shape of the data
df.shape

# Removing columns
list_drop = ['cp', 'chol', 'fbs', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
df.drop(list_drop, axis=1, inplace=True)

# Print the data after removing columns
df.head()

# Find a duplicate rows in dataset
duplicate_data_rows = df[df.duplicated(keep='first')]
print(duplicate_data_rows)

# Drop duplicate rows in dataset
new_df = df.drop_duplicates(keep='first')
new_df.reset_index(inplace=True)
del new_df['index']
new_df.shape

# Print dataset
new_df

# Count the empty values in each column in dataset
new_df.isna().sum()

# View basic statistics
new_df.describe()

# Visualize the count of the number of patients with a heart disease and without.
sns.countplot(x = 'target', data = new_df, palette = 'RdBu_r')

# Get the correlation in data set.
new_df.corr()

# Get correlations of each features in dataset
corr_relationsmat = new_df.corr()
top_corr_relations_features = corr_relationsmat.index
plt.figure(figsize=(10,10))

# Visualize the data
sns.heatmap(new_df[top_corr_relations_features].corr(), annot=True)

# Spit into feature data and target data.
X = new_df.loc[:, new_df.columns != 'target']
Y = new_df['target']

# Check number of heart attacks compared to gender:
sns.countplot(x = 'target', data = new_df, hue = 'sex')
plt.title("Heart Disease Frequency for Gender")
plt.xlabel("0 = No heart Disease, 1 = Heart Disease");

# Creating Histograms for dataset
new_df.hist(figsize=(15,10))

# Split into 80% training data and 20% testing dataset.
from sklearn.model_selection import train_test_split
X_train_dataset, X_test_dataset, Y_train_dataset, Y_test_dataset = train_test_split(X, Y, test_size= 0.20, random_state = 1)

# Use RandomForestClassifier.
from sklearn.ensemble import RandomForestClassifier
randomForestClassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 1)

# Fit model on training data
randomForestClassifier.fit(X_train_dataset, Y_train_dataset)

from sklearn.metrics import accuracy_score
# Test the models accuracy on the data set.
heart_attack_random_forest_model = randomForestClassifier

# Predict accuracy
Y_dataset_rf_accuracy = heart_attack_random_forest_model.predict(X_test_dataset)
model_accuracy_score_rf = round(accuracy_score(Y_dataset_rf_accuracy, Y_test_dataset) * 100 , 2)

# Print Model test Accuracy
print("Model Test Accuracy: " + str(model_accuracy_score_rf)+" %")

import joblib
# Save model in 'IdentifyHeartAttackModel' directory
joblib.dump(heart_attack_random_forest_model,'./IdentifyHeartAttackModel/identify_heart_attack_randomforest_model.joblib')