import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
dataset = pd.read_csv('Telco-Customer-Churn.csv')

# 2. Explore the dataset
print("First 5 rows:")
print(dataset.head())

print("\nMissing values in each column:")
print(dataset.isnull().sum())

print("\nDataset statistical summary:")
print(dataset.describe())

# 3. Check churn distribution
print("\nChurn distribution:")
print(dataset['Churn'].value_counts())

# Plot churn distribution
sns.countplot(
    x='Churn',
    hue='Churn',
    data=dataset,
    palette='coolwarm',
    legend=False
)

plt.title('Churn Distribution')
plt.xlabel('Churn (No = 0, Yes = 1)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 4. Convert TotalCharges to numeric
dataset['TotalCharges'] = pd.to_numeric(
    dataset['TotalCharges'],
    errors='coerce'
)

# 5. Handle missing values in TotalCharges
dataset['TotalCharges'] = dataset['TotalCharges'].fillna(
    dataset['TotalCharges'].median()
)

# 6. Encode categorical columns
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
categorical_cols = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'Churn'
]

for col in categorical_cols:
    dataset[col] = labelencoder.fit_transform(dataset[col].astype(str))

# 7. Display the processed dataset
print("\nProcessed dataset:")
print(dataset.head())

print("\nDataset information after preprocessing:")
print(dataset.info())

print("\nMissing values after preprocessing:")
print(dataset.isnull().sum())


# 8. Feature Selection and Splitting Data
from sklearn.model_selection import train_test_split

X = dataset.drop(['customerID', 'Churn'], axis=1)
y = dataset['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 9. Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 10. Model Training and Prediction
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


# 11. Model Evaluation: Accuracy Score

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 11. Model Evaluation: Confusion Matrix and Performance Metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="coolwarm")
plt.title('Confusion Matrix')
plt.show()



