# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Load the dataset
# Define the data types of the columns in the CSV file
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)

# Define the column names
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df.replace("?", "unknown", inplace=True)
print(df)
# Encode categorical features (Outlook, Temp, Humid, Wind) into numerical values
label_encoders = {}
for column in df.columns:
    le = preprocessing.LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

print(df)


# Convert the target variable to binary
#df['income'] = df['income'].apply(lambda x: 0 if x == ' <=50K' else 1)
# Get a list of all the numeric columns in the dataset

X = df.drop('income', axis=1)
y = df['income']
print(X)
print(y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mtry = 10
rf = RandomForestClassifier(n_estimators=100,max_features=mtry, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='mtry = %d, AUC = %0.2f' % (mtry, roc_auc))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with mtry={}: {:.2f}".format(mtry, accuracy))