
# ============================================================
# LOAN APPROVAL PREDICTION USING SUPPORT VECTOR MACHINE (SVM)
# ============================================================


# ============================================================
# 1. Import Required Libraries
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings("ignore")



# ============================================================
# 2. Load the Dataset
# ============================================================

# Get the folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Dataset path
file_path = BASE_DIR / "loan_data.csv"

# Check whether the dataset exists
if not file_path.exists():
    raise FileNotFoundError(
        "loan_data.csv was not found.\n"
        "Place the CSV file in the same folder as this Python file."
    )

# Load dataset
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# Display first five records
print("\nFirst Five Records:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display dataset information
print("\nDataset Information:")
df.info()

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe(include="all"))

# Check required target column
if "Loan_Status" not in df.columns:
    raise ValueError(
        "The dataset must contain a Loan_Status column."
    )

# Remove records with missing target values
df = df.dropna(subset=["Loan_Status"]).copy()

# Remove unique identifier if available
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)

if df.empty:
    raise ValueError("No valid records found in the dataset.")

# ============================================================
# 3. Handle Missing Values
# ============================================================

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

numerical_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

# Fill missing categorical values using mode
for col in categorical_cols:

    if df[col].isnull().any():

        mode_value = df[col].mode()

        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value.iloc[0])


# Fill missing numerical values using median
for col in numerical_cols:

    if df[col].isnull().any():

        median_value = df[col].median()

        df[col] = df[col].fillna(median_value)


# Check missing values
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Remove any remaining rows with missing values
df = df.dropna().copy()

if df.empty:
    raise ValueError("No records remain after handling missing values.")


# ============================================================
# 4. Visualize Loan Approval Distribution
# ============================================================

temp = df["Loan_Status"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    temp.values,
    labels=temp.index.astype(str),
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Loan Approval Status Distribution")

plt.tight_layout()
plt.show()


# ============================================================
# 5. Visualize Gender and Marital Status
# ============================================================

categorical_plot_cols = [
    col for col in ["Gender", "Married"]
    if col in df.columns
]

if categorical_plot_cols:

    number_of_columns = len(categorical_plot_cols)

    fig, axes = plt.subplots(
        1,
        number_of_columns,
        figsize=(7 * number_of_columns, 5)
    )

    axes = np.array(axes).flatten()

    for index, col in enumerate(categorical_plot_cols):

        sb.countplot(
            data=df,
            x=col,
            hue="Loan_Status",
            ax=axes[index]
        )

        axes[index].set_title(
            f"{col} vs Loan Approval Status"
        )

        axes[index].set_xlabel(col)
        axes[index].set_ylabel("Count")

        axes[index].tick_params(
            axis="x",
            rotation=45
        )

    plt.tight_layout()
    plt.show()

else:
    print("\nGender and Married columns are unavailable.")


# ============================================================
# 6. Visualize Numerical Variables Using Histograms
# ============================================================

numerical_plot_cols = [
    col for col in ["ApplicantIncome", "LoanAmount"]
    if col in df.columns
]

if numerical_plot_cols:

    fig, axes = plt.subplots(
        1,
        len(numerical_plot_cols),
        figsize=(7 * len(numerical_plot_cols), 5)
    )

    axes = np.array(axes).flatten()

    for index, col in enumerate(numerical_plot_cols):

        axes[index].hist(
            df[col],
            bins=30,
            edgecolor="black"
        )

        axes[index].set_title(
            f"Distribution of {col}"
        )

        axes[index].set_xlabel(col)
        axes[index].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

else:
    print("\nRequired numerical columns are unavailable.")


# ============================================================
# 7. Visualize Numerical Variables Using Boxplots
# ============================================================

if numerical_plot_cols:

    fig, axes = plt.subplots(
        1,
        len(numerical_plot_cols),
        figsize=(7 * len(numerical_plot_cols), 5)
    )

    axes = np.array(axes).flatten()

    for index, col in enumerate(numerical_plot_cols):

        sb.boxplot(
            x=df[col],
            ax=axes[index]
        )

        axes[index].set_title(
            f"Boxplot of {col}"
        )

        axes[index].set_xlabel(col)

    plt.tight_layout()
    plt.show()


# ============================================================
# 8. Remove Extreme Values
# ============================================================

# Apply the original case study's filtering conditions

if "ApplicantIncome" in df.columns:

    df = df[
        df["ApplicantIncome"] < 25000
    ].copy()

if "LoanAmount" in df.columns:

    df = df[
        df["LoanAmount"] < 400000
    ].copy()

if df.empty:
    raise ValueError(
        "No records remain after outlier filtering. "
        "Review the filtering thresholds."
    )

print("\nDataset Shape After Outlier Removal:")
print(df.shape)


# ============================================================
# 9. Analyze Loan Amount by Gender
# ============================================================

if all(
    col in df.columns
    for col in ["Gender", "LoanAmount"]
):

    print("\nAverage Loan Amount by Gender:")

    print(
        df.groupby("Gender")["LoanAmount"].mean()
    )


# ============================================================
# 10. Analyze Loan Amount by Marital Status and Gender
# ============================================================

if all(
    col in df.columns
    for col in ["Married", "Gender", "LoanAmount"]
):

    print("\nAverage Loan Amount by Married Status and Gender:")

    print(
        df.groupby(
            ["Married", "Gender"]
        )["LoanAmount"].mean()
    )


# ============================================================
# 11. Encode Categorical Variables
# ============================================================

# Separate target variable
target = df["Loan_Status"].astype(str).copy()

# Separate input features
features = df.drop(
    columns=["Loan_Status"]
).copy()

# Identify categorical feature columns
categorical_cols = features.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Store label encoders
label_encoders = {}

# Encode categorical features
for col in categorical_cols:

    label_encoder = LabelEncoder()

    features[col] = label_encoder.fit_transform(
        features[col].astype(str)
    )

    label_encoders[col] = label_encoder


# Encode target variable
target_encoder = LabelEncoder()

target = target_encoder.fit_transform(target)

print("\nCategorical Encoding Completed!")

print("\nEncoded Feature Dataset:")
print(features.head())

print("\nLoan Status Classes:")
print(target_encoder.classes_)

# Check that the target has exactly two classes
if len(np.unique(target)) != 2:
    raise ValueError(
        "This SVM ROC-AUC case study requires exactly "
        "two Loan_Status classes."
    )

# Check feature data types
if not all(
    pd.api.types.is_numeric_dtype(dtype)
    for dtype in features.dtypes
):
    raise ValueError(
        "Some categorical features were not encoded."
    )


# ============================================================
# 12. Generate Correlation Heatmap
# ============================================================

correlation_data = features.copy()

correlation_data["Loan_Status"] = target

correlation_matrix = correlation_data.corr()

plt.figure(figsize=(12, 8))

sb.heatmap(
    correlation_matrix > 0.8,
    annot=True,
    cbar=False,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 13. Separate Features and Target Variables
# ============================================================

X = features

Y = target

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(Y.shape)


# ============================================================
# 14. Split Training and Validation Data
# ============================================================

# 80% Training Data
# 20% Validation Data

class_counts = pd.Series(Y).value_counts()

if class_counts.min() < 2:
    raise ValueError(
        "Each Loan_Status class must have at least "
        "two records for stratified splitting."
    )

X_train, X_val, Y_train, Y_val = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=10,
    stratify=Y
)

print("\nTraining and Validation Split Completed!")

print("X_train Shape:", X_train.shape)
print("X_val Shape:", X_val.shape)

print("Y_train Shape:", Y_train.shape)
print("Y_val Shape:", Y_val.shape)


# ============================================================
# 15. Balance the Training Dataset
# ============================================================

# Check class distribution before oversampling
print("\nTraining Class Distribution Before Oversampling:")

print(pd.Series(Y_train).value_counts())

# Apply Random Over Sampling
ros = RandomOverSampler(
    sampling_strategy="minority",
    random_state=0
)

X_train_balanced, Y_train_balanced = ros.fit_resample(
    X_train,
    Y_train
)

# Check class distribution after oversampling
print("\nTraining Class Distribution After Oversampling:")

print(pd.Series(Y_train_balanced).value_counts())

print("\nOriginal Training Shape:", X_train.shape)

print("Balanced Training Shape:", X_train_balanced.shape)


# ============================================================
# 16. Normalize the Features
# ============================================================

# StandardScaler standardizes features
# using the training dataset.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_balanced
)

X_val_scaled = scaler.transform(
    X_val
)

print("\nFeature Scaling Completed!")


# ============================================================
# 17. Initialize Support Vector Machine Model
# ============================================================

model = SVC(
    kernel="rbf",
    probability=True,
    random_state=10
)

print("\nSupport Vector Machine Model Initialized!")


# ============================================================
# 18. Train the SVM Model
# ============================================================

model.fit(
    X_train_scaled,
    Y_train_balanced
)

print("\nSVM Model Training Completed Successfully!")


# ============================================================
# 19. Predict Loan Approval Status
# ============================================================

# Predictions on balanced training data
Y_train_pred = model.predict(
    X_train_scaled
)

# Predictions on validation data
Y_val_pred = model.predict(
    X_val_scaled
)

# Prediction probabilities
Y_train_prob = model.predict_proba(
    X_train_scaled
)[:, 1]

Y_val_prob = model.predict_proba(
    X_val_scaled
)[:, 1]


# ============================================================
# 20. Evaluate Model Performance
# ============================================================

# Training Accuracy
training_accuracy = accuracy_score(
    Y_train_balanced,
    Y_train_pred
)

# Validation Accuracy
validation_accuracy = accuracy_score(
    Y_val,
    Y_val_pred
)

# Training ROC-AUC
training_roc_auc = roc_auc_score(
    Y_train_balanced,
    Y_train_prob
)

# Validation ROC-AUC
validation_roc_auc = roc_auc_score(
    Y_val,
    Y_val_prob
)

print("\n======================================")

print("SVM MODEL PERFORMANCE")

print("======================================")

print(
    "Training Accuracy:",
    f"{training_accuracy * 100:.2f}%"
)

print(
    "Validation Accuracy:",
    f"{validation_accuracy * 100:.2f}%"
)

print(
    "Training ROC-AUC Score:",
    f"{training_roc_auc:.4f}"
)

print(
    "Validation ROC-AUC Score:",
    f"{validation_roc_auc:.4f}"
)


# ============================================================
# 21. Generate Confusion Matrix
# ============================================================

cm = confusion_matrix(
    Y_val,
    Y_val_pred
)

plt.figure(figsize=(6, 6))

sb.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=target_encoder.classes_,
    yticklabels=target_encoder.classes_
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Loan Status")

plt.ylabel("Actual Loan Status")

plt.tight_layout()
plt.show()


# ============================================================
# 22. Generate Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        Y_val,
        Y_val_pred,
        labels=[0, 1],
        target_names=target_encoder.classes_.astype(str),
        zero_division=0
    )
)


# ============================================================
# 23. Final Results
# ============================================================

print("\n======================================")

print("LOAN APPROVAL PREDICTION COMPLETED")

print("======================================")

print(
    "Final Validation Accuracy:",
    f"{validation_accuracy * 100:.2f}%"
)

print(
    "Final Validation ROC-AUC:",
    f"{validation_roc_auc:.4f}"
)

print("\nModel evaluation completed successfully!")

