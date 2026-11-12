
# ============================================================
# ONLINE ADVERTISEMENT CLICK PREDICTION USING XGBOOST
# ============================================================


# 1. Import Required Libraries

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from xgboost import XGBClassifier


# ============================================================
# 2. Load the Dataset
# ============================================================

# Get the folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Dataset path
file_path = BASE_DIR / "ad_10000records.csv"

# Check whether the dataset exists
if not file_path.exists():
    raise FileNotFoundError(
        "Dataset not found: ad_10000records.csv\n"
        "Please place the CSV file in the same folder as this script."
    )

# Load the CSV dataset
data = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# Display first five records
print("\nFirst Five Records:")
print(data.head())

# Display dataset shape
print("\nDataset Shape:")
print(data.shape)

# Display column names
print("\nDataset Columns:")
print(data.columns.tolist())

# Display dataset information
print("\nDataset Information:")
data.info()

# ============================================================
# 3. Check and Clean the Dataset
# ============================================================

# Required columns
required_columns = [
    "Gender",
    "Ad Topic Line",
    "City",
    "Clicked on Ad"
]

# Check whether required columns exist
missing_columns = [
    col for col in required_columns if col not in data.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: " + ", ".join(missing_columns)
    )

# Remove rows containing missing values
data = data.dropna().copy()

print("\nDataset Shape After Cleaning:")
print(data.shape)

if data.empty:
    raise ValueError("The dataset is empty after removing missing values.")


# ============================================================
# 4. Analyze Advertisement Clicks
# ============================================================

print("\nAdvertisement Click Distribution:")

click_counts = data["Clicked on Ad"].value_counts()

print(click_counts)


# ============================================================
# 5. Calculate Click-Through Rate (CTR)
# ============================================================

# CTR = (Total Clicks / Total Impressions) * 100

total_records = len(data)

total_clicks = (data["Clicked on Ad"] == 1).sum()

click_through_rate = (
    total_clicks / total_records
) * 100

print("\nClick-Through Rate Analysis:")

print("Total Records:", total_records)

print("Total Clicks:", total_clicks)

print(f"Click Through Rate: {click_through_rate:.2f}%")


# ============================================================
# 6. Encode the Gender Column
# ============================================================

# Convert categorical Gender values into numerical values

le = LabelEncoder()

data["Gender"] = le.fit_transform(
    data["Gender"].astype(str)
)

print("\nGender Encoding Completed!")

print("\nGender Categories:")

for category, encoded_value in zip(
    le.classes_,
    le.transform(le.classes_)
):
    print(category, "=", encoded_value)


# ============================================================
# 7. Select Input Features
# ============================================================

# Following the original case study:
# Select the first seven columns as input features

x = data.iloc[:, 0:7].copy()

# Remove categorical text columns
x = x.drop(
    columns=["Ad Topic Line", "City"],
    errors="ignore"
)

# Convert remaining feature columns into numeric values
for column in x.columns:
    x[column] = pd.to_numeric(
        x[column],
        errors="coerce"
    )

# Check for invalid numerical values
if x.isnull().any().any():
    raise ValueError(
        "Some selected feature columns contain non-numeric values. "
        "Please check the dataset."
    )

print("\nSelected Input Features:")

print(x.head())

print("\nFeature Columns:")

print(x.columns.tolist())


# ============================================================
# 8. Select Target Variable
# ============================================================

# Target variable: Clicked on Ad

y = pd.to_numeric(
    data["Clicked on Ad"],
    errors="coerce"
)

if y.isnull().any():
    raise ValueError(
        "The target column contains invalid values."
    )

# Ensure target contains binary values
if not set(y.unique()).issubset({0, 1}):
    raise ValueError(
        "The target column 'Clicked on Ad' must contain only 0 and 1."
    )

y = y.astype(int)

print("\nTarget Variable Distribution:")

print(y.value_counts())


# ============================================================
# 9. Split Training and Testing Data
# ============================================================

# 80% Training Data
# 20% Testing Data

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining and Testing Split Completed!")

print("Training Features:", x_train.shape)

print("Testing Features:", x_test.shape)

print("Training Target:", y_train.shape)

print("Testing Target:", y_test.shape)


# ============================================================
# 10. Initialize XGBoost Classifier
# ============================================================

model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

print("\nXGBoost Classifier Initialized!")


# ============================================================
# 11. Train the Model
# ============================================================

model.fit(
    x_train,
    y_train
)

print("\nModel Training Completed!")


# ============================================================
# 12. Predict Advertisement Clicks
# ============================================================

y_pred = model.predict(x_test)

print("\nPredictions Completed!")


# ============================================================
# 13. Evaluate Model Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Evaluation:")

print("The model accuracy is:", accuracy)

print(f"Accuracy Percentage: {accuracy * 100:.2f}%")


# ============================================================
# 14. Generate Confusion Matrix
# ============================================================

conf_matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")

print(conf_matrix)


# ============================================================
# 15. Final Results
# ============================================================

print("\n====================================")

print("ADVERTISEMENT CLICK PREDICTION")

print("====================================")

print("Total Records:", total_records)

print("Total Advertisement Clicks:", total_clicks)

print(f"Click Through Rate: {click_through_rate:.2f}%")

print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nCase Study Completed Successfully!")

