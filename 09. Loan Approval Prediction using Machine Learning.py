
# ============================================================
# LOAN APPROVAL PREDICTION USING MACHINE LEARNING
# ============================================================


# 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn import metrics

# ============================================================
# 2. Load the Dataset
# ============================================================

# Get the folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Dataset path
file_path = BASE_DIR / "LoanApprovalPrediction.csv"

# Check whether the dataset exists
if not file_path.exists():
    raise FileNotFoundError(
        "LoanApprovalPrediction.csv was not found.\n"
        "Place the CSV file in the same folder as this Python file."
    )

# Load the dataset
data = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# Display first five records
print("\nFirst Five Records:")
print(data.head(5))

# Display dataset shape
print("\nDataset Shape:")
print(data.shape)

# Display dataset information
print("\nDataset Information:")
data.info()

# Display missing values
print("\nMissing Values:")
print(data.isnull().sum())


# ============================================================
# 3. Identify Categorical Variables
# ============================================================

obj = (data.dtypes == "object")

object_cols = list(obj[obj].index)

print("\nCategorical Variables:", len(object_cols))

print("Categorical Columns:")
print(object_cols)


# ============================================================
# 4. Drop Loan_ID Column
# ============================================================

# Loan_ID is a unique identifier and is not useful
# as a predictive feature.

if "Loan_ID" in data.columns:
    data.drop(["Loan_ID"], axis=1, inplace=True)

print("\nLoan_ID column removed.")

print("\nUpdated Dataset:")
print(data.head())



# ============================================================
# 5. Visualize Categorical Variables
# ============================================================

# Identify categorical columns
object_cols = data.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("\nCategorical Columns Found:")
print(object_cols)

# Check whether categorical columns exist
if len(object_cols) == 0:

    print("No categorical columns found to visualize.")

else:

    # Calculate subplot rows
    number_of_columns = 4

    number_of_rows = int(
        np.ceil(len(object_cols) / number_of_columns)
    )

    # Create figure
    fig, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(18, 5 * number_of_rows)
    )

    # Convert axes into a one-dimensional array
    axes = np.array(axes).flatten()

    # Visualize each categorical column
    for index, col in enumerate(object_cols):

        # Count categories
        category_counts = data[col].value_counts(
            dropna=False
        )

        # Create bar chart
        axes[index].bar(
            category_counts.index.astype(str),
            category_counts.values
        )

        axes[index].set_title(col)

        axes[index].set_xlabel(col)

        axes[index].set_ylabel("Count")

        axes[index].tick_params(
            axis="x",
            rotation=45
        )

    # Hide unused subplots
    for index in range(len(object_cols), len(axes)):
        axes[index].set_visible(False)

    plt.suptitle(
        "Distribution of Categorical Variables",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()

# ============================================================
# 6. Handle Missing Values
# ============================================================

# Separate categorical and numerical columns

categorical_cols = data.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_cols = data.select_dtypes(
    include=[np.number]
).columns.tolist()


# Fill missing categorical values using mode

for col in categorical_cols:

    if data[col].isnull().any():

        mode_value = data[col].mode()

        if not mode_value.empty:

            data[col] = data[col].fillna(mode_value.iloc[0])


# Fill missing numerical values using median

for col in numerical_cols:

    if data[col].isnull().any():

        median_value = data[col].median()

        data[col] = data[col].fillna(median_value)


# Check missing values after cleaning

print("\nMissing Values After Cleaning:")

print(data.isnull().sum())

# Remove rows with missing target values, if any
data = data.dropna(subset=["Loan_Status"]).copy()

if data.empty:
    raise ValueError("No valid records remain in the dataset.")



# ============================================================
# 7. Encode Categorical Variables
# ============================================================

# Identify categorical columns
categorical_cols = data.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Store label encoders
label_encoders = {}

# Encode categorical columns
for col in categorical_cols:

    label_encoder = LabelEncoder()

    data[col] = label_encoder.fit_transform(
        data[col].astype(str)
    )

    label_encoders[col] = label_encoder

print("\nCategorical Encoding Completed!")

print("\nEncoded Dataset:")
print(data.head())

# Check remaining non-numerical columns
remaining_cols = data.select_dtypes(
    exclude=[np.number]
).columns.tolist()

print("\nRemaining Non-Numerical Columns:")
print(remaining_cols)

if remaining_cols:
    raise ValueError(
        f"Categorical columns were not encoded: {remaining_cols}"
    )




# ============================================================
# 8. Generate Correlation Heatmap
# ============================================================

# Select numerical columns only
numerical_data = data.select_dtypes(include=[np.number])

# Calculate correlation matrix
correlation_matrix = numerical_data.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    cmap="BrBG",
    fmt=".2f",
    linewidths=2,
    annot=True
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 9. Analyze Gender, Marital Status and Loan Status
# ============================================================

# Create a bar chart to understand the relationship
# between Gender, Married and Loan_Status.

if all(
    col in data.columns
    for col in ["Gender", "Married", "Loan_Status"]
):

    sns.catplot(
        x="Gender",
        y="Married",
        hue="Loan_Status",
        kind="bar",
        data=data,
        height=5,
        aspect=1.3
    )

    plt.title("Loan Status by Gender and Marital Status")

    plt.show()

else:
    print(
        "\nGender, Married or Loan_Status column "
        "is unavailable for this visualization."
    )


# ============================================================
# 10. Separate Features and Target Variables
# ============================================================

# Input features
X = data.drop(["Loan_Status"], axis=1)

# Target variable
Y = data["Loan_Status"]

print("\nFeature Shape:", X.shape)

print("Target Shape:", Y.shape)


# Check whether all input features are numerical
print("\nFeature Data Types:")
print(X.dtypes)

if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
    raise ValueError("All input features must be numerical before training.")

# ============================================================
# 11. Split Training and Testing Data
# ============================================================

# 60% Training Data
# 40% Testing Data

# Use stratification to preserve class proportions
# when both classes have enough records.

if Y.value_counts().min() >= 2:

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.4,
        random_state=1,
        stratify=Y
    )

else:

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.4,
        random_state=1
    )


print("\nTraining and Testing Split Completed!")

print("X_train Shape:", X_train.shape)

print("X_test Shape:", X_test.shape)

print("Y_train Shape:", Y_train.shape)

print("Y_test Shape:", Y_test.shape)


# ============================================================
# 12. Initialize Machine Learning Models
# ============================================================

# K-Nearest Neighbors
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=3))
])


# Random Forest Classifier
rfc = RandomForestClassifier(
    n_estimators=7,
    criterion="entropy",
    random_state=7
)


# Support Vector Machine
svc = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC())
])


# Logistic Regression
lc = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Store models in a list
models = [rfc, knn, svc, lc]

print("\nMachine Learning Models Initialized!")


# ============================================================
# 13. Train the Models
# ============================================================

for clf in models:

    clf.fit(X_train, Y_train)

    print(
        "\nTraining Completed:",
        clf.__class__.__name__
    )


# ============================================================
# 14. Predict Loan Approval Status
# ============================================================

# Compare training and testing accuracy

results = []

for clf in models:

    # Training predictions
    Y_train_pred = clf.predict(X_train)

    # Testing predictions
    Y_test_pred = clf.predict(X_test)

    # Calculate training accuracy
    train_accuracy = metrics.accuracy_score(
        Y_train,
        Y_train_pred
    )

    # Calculate testing accuracy
    test_accuracy = metrics.accuracy_score(
        Y_test,
        Y_test_pred
    )

    # Store model results
    results.append({
        "Model": clf.__class__.__name__,
        "Training Accuracy": train_accuracy * 100,
        "Testing Accuracy": test_accuracy * 100
    })

    print("\nModel:", clf.__class__.__name__)

    print(
        "Training Accuracy:",
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        "Testing Accuracy:",
        f"{test_accuracy * 100:.2f}%"
    )


# ============================================================
# 15. Compare Model Performance
# ============================================================

results_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")

print(results_df.to_string(index=False))


# ============================================================
# 16. Visualize Model Accuracy
# ============================================================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Testing Accuracy"
)

plt.title("Machine Learning Model Accuracy Comparison")

plt.xlabel("Machine Learning Models")

plt.ylabel("Testing Accuracy (%)")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()


# ============================================================
# 17. Final Results
# ============================================================

best_model = results_df.loc[
    results_df["Testing Accuracy"].idxmax()
]

print("\n======================================")

print("LOAN APPROVAL PREDICTION RESULTS")

print("======================================")

print("Best Performing Model:")

print(best_model["Model"])

print(
    "Testing Accuracy:",
    f"{best_model['Testing Accuracy']:.2f}%"
)

print("\nLoan Approval Prediction Completed Successfully!")

