# ============================================================
# HOUSE PRICE PREDICTION USING MACHINE LEARNING
# ============================================================


# ============================================================
# 1. Import Required Libraries
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 2. Load the Dataset
# ============================================================

# Get the folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Dataset path
file_path = BASE_DIR / "HousePricePrediction.xlsx"

# Check whether the dataset exists
if not file_path.exists():
    raise FileNotFoundError(
        "HousePricePrediction.xlsx was not found.\n"
        "Place the Excel file in the same folder as this Python file."
    )

# Load Excel dataset
dataset = pd.read_excel(file_path)

print("Dataset loaded successfully!")

# Display first five records
print("\nFirst Five Records:")
print(dataset.head(5))

# Display dataset shape
print("\nDataset Shape:")
print(dataset.shape)

# Display dataset information
print("\nDataset Information:")
dataset.info()

# Display statistical summary
print("\nStatistical Summary:")
print(dataset.describe(include="all"))


# ============================================================
# 3. Identify Categorical, Integer and Float Variables
# ============================================================

# Identify categorical columns
object_cols = dataset.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Identify integer columns
integer_cols = dataset.select_dtypes(
    include=["int64", "int32"]
).columns.tolist()

# Identify floating-point columns
float_cols = dataset.select_dtypes(
    include=["float64", "float32"]
).columns.tolist()

print("\nCategorical Variables:", len(object_cols))
print("Categorical Columns:", object_cols)

print("\nInteger Variables:", len(integer_cols))
print("Integer Columns:", integer_cols)

print("\nFloat Variables:", len(float_cols))
print("Float Columns:", float_cols)


# ============================================================
# 4. Generate Correlation Heatmap
# ============================================================

# Select numerical columns
numerical_dataset = dataset.select_dtypes(
    include=[np.number]
)

if not numerical_dataset.empty:

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        numerical_dataset.corr(),
        cmap="BrBG",
        fmt=".2f",
        linewidths=1,
        annot=True
        if numerical_dataset.shape[1] <= 15
        else False
    )

    plt.title("Correlation Heatmap of Numerical Features")

    plt.tight_layout()

    # Save heatmap
    heatmap_path = BASE_DIR / "correlation_heatmap.png"

    plt.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nHeatmap saved:", heatmap_path)

    plt.show()

else:
    print("\nNo numerical columns available for correlation heatmap.")


# ============================================================
# 5. Visualize Number of Unique Categorical Values
# ============================================================

if object_cols:

    unique_values = []

    for col in object_cols:

        unique_values.append(
            dataset[col].nunique(dropna=False)
        )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=object_cols,
        y=unique_values
    )

    plt.title(
        "Number of Unique Values in Categorical Features"
    )

    plt.xlabel("Categorical Features")

    plt.ylabel("Number of Unique Values")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.show()

else:
    print("\nNo categorical variables found.")



# ============================================================
# 6. Visualize Categorical Feature Distributions
# ============================================================

# Identify categorical columns
object_cols = dataset.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

if object_cols:

    # Calculate subplot dimensions
    number_of_columns = 4

    number_of_rows = int(
        np.ceil(len(object_cols) / number_of_columns)
    )

    # Create subplots
    fig, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(18, 5 * number_of_rows)
    )

    # Convert axes into a one-dimensional array
    axes = np.array(axes).flatten()

    # Visualize each categorical column
    for index, col in enumerate(object_cols):

        # Convert all values into consistent strings
        column_data = dataset[col].astype("string").fillna("Missing")

        # Count category occurrences
        category_counts = column_data.value_counts().head(15)

        # Convert labels into strings
        category_labels = [
            str(value) for value in category_counts.index
        ]

        # Convert counts into numerical values
        category_values = category_counts.values.astype(int)

        # Use numerical positions instead of categorical axis values
        x_positions = np.arange(len(category_labels))

        # Create bar chart
        axes[index].bar(
            x_positions,
            category_values
        )

        # Set category labels
        axes[index].set_xticks(x_positions)

        axes[index].set_xticklabels(
            category_labels,
            rotation=90
        )

        axes[index].set_title(
            f"Distribution of {col}"
        )

        axes[index].set_xlabel(col)

        axes[index].set_ylabel("Count")

    # Hide unused subplot spaces
    for index in range(len(object_cols), len(axes)):

        axes[index].set_visible(False)

    plt.suptitle(
        "Categorical Features: Distribution",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()

else:

    print(
        "\nNo categorical features available for visualization."
    )



# ============================================================
# 7. Drop Unnecessary Columns
# ============================================================

# Id is a unique identifier, not a useful predictive feature.
if "Id" in dataset.columns:

    dataset.drop(
        columns=["Id"],
        inplace=True
    )

    print("\nId column removed.")


# Check whether SalePrice exists
if "SalePrice" not in dataset.columns:

    raise ValueError(
        "The dataset must contain a SalePrice target column."
    )


# ============================================================
# 8. Handle Missing Target Values
# ============================================================

# Convert SalePrice to numerical format
dataset["SalePrice"] = pd.to_numeric(
    dataset["SalePrice"],
    errors="coerce"
)

# Remove records with missing or invalid SalePrice
dataset.dropna(
    subset=["SalePrice"],
    inplace=True
)

# Remove records with invalid or non-positive house prices
dataset = dataset[
    dataset["SalePrice"] > 0
].copy()

if dataset.empty:

    raise ValueError(
        "No valid SalePrice records remain in the dataset."
    )

print("\nDataset Shape After Target Cleaning:")
print(dataset.shape)


# ============================================================
# 9. Remove Completely Empty Features
# ============================================================

# Remove columns containing only missing values
empty_cols = dataset.columns[
    dataset.isnull().all()
].tolist()

if empty_cols:

    dataset.drop(
        columns=empty_cols,
        inplace=True
    )

    print("\nCompletely Empty Columns Removed:")
    print(empty_cols)


# ============================================================
# 10. Check Missing Values
# ============================================================

print("\nMissing Values in Each Column:")

print(dataset.isnull().sum())

print("\nTotal Missing Values:")

print(dataset.isnull().sum().sum())


# ============================================================
# 11. Identify Categorical Variables for Encoding
# ============================================================

# Separate target and input features
X = dataset.drop(
    columns=["SalePrice"]
).copy()

Y = dataset["SalePrice"].copy()

# Identify categorical columns
object_cols = X.select_dtypes(
    include=["object", "str", "category", "bool"]
).columns.tolist()

print("\nCategorical Variables:")

print(object_cols)

print(
    "\nNumber of Categorical Features:",
    len(object_cols)
)


# Identify numerical columns
numerical_cols = X.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nNumerical Variables:")

print(numerical_cols)

if not numerical_cols and not object_cols:

    raise ValueError(
        "No usable input features were found."
    )


# ============================================================
# 12. Split Training and Validation Data
# ============================================================

# 80% Training Data
# 20% Validation Data

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X,
    Y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)

print("\nTraining and Validation Split Completed!")

print("X_train Shape:", X_train.shape)

print("X_valid Shape:", X_valid.shape)

print("Y_train Shape:", Y_train.shape)

print("Y_valid Shape:", Y_valid.shape)


# ============================================================
# 13. Prepare Data Preprocessing
# ============================================================

# Create preprocessing steps for numerical features
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Create preprocessing steps for categorical features
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, object_cols)
    ],
    remainder="drop"
)

print("\nData Preprocessing Pipeline Created!")


# ============================================================
# 14. Initialize Support Vector Regression Model
# ============================================================

model_SVR = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", SVR())
    ]
)

print("\nSupport Vector Regression Model Initialized!")


# ============================================================
# 15. Train Support Vector Regression Model
# ============================================================

model_SVR.fit(
    X_train,
    Y_train
)

print("\nSVR Model Training Completed!")

# Predict house prices
Y_pred_SVR = model_SVR.predict(
    X_valid
)


# ============================================================
# 16. Evaluate Support Vector Regression Model
# ============================================================

mae_SVR = mean_absolute_error(
    Y_valid,
    Y_pred_SVR
)

mape_SVR = mean_absolute_percentage_error(
    Y_valid,
    Y_pred_SVR
)

r2_SVR = r2_score(
    Y_valid,
    Y_pred_SVR
)

print("\nSupport Vector Regression Results:")

print("Mean Absolute Error:", f"{mae_SVR:.2f}")

print("Mean Absolute Percentage Error:", f"{mape_SVR * 100:.2f}%")

print("R2 Score:", f"{r2_SVR:.4f}")


# ============================================================
# 17. Initialize Random Forest Regression Model
# ============================================================

model_RFR = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

print("\nRandom Forest Regression Model Initialized!")


# ============================================================
# 18. Train Random Forest Regression Model
# ============================================================

model_RFR.fit(
    X_train,
    Y_train
)

print("\nRandom Forest Model Training Completed!")

# Predict house prices
Y_pred_RFR = model_RFR.predict(
    X_valid
)


# ============================================================
# 19. Evaluate Random Forest Regression Model
# ============================================================

mae_RFR = mean_absolute_error(
    Y_valid,
    Y_pred_RFR
)

mape_RFR = mean_absolute_percentage_error(
    Y_valid,
    Y_pred_RFR
)

r2_RFR = r2_score(
    Y_valid,
    Y_pred_RFR
)

print("\nRandom Forest Regression Results:")

print("Mean Absolute Error:", f"{mae_RFR:.2f}")

print("Mean Absolute Percentage Error:", f"{mape_RFR * 100:.2f}%")

print("R2 Score:", f"{r2_RFR:.4f}")


# ============================================================
# 20. Initialize Linear Regression Model
# ============================================================

model_LR = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

print("\nLinear Regression Model Initialized!")


# ============================================================
# 21. Train Linear Regression Model
# ============================================================

model_LR.fit(
    X_train,
    Y_train
)

print("\nLinear Regression Model Training Completed!")

# Predict house prices
Y_pred_LR = model_LR.predict(
    X_valid
)


# ============================================================
# 22. Evaluate Linear Regression Model
# ============================================================

mae_LR = mean_absolute_error(
    Y_valid,
    Y_pred_LR
)

mape_LR = mean_absolute_percentage_error(
    Y_valid,
    Y_pred_LR
)

r2_LR = r2_score(
    Y_valid,
    Y_pred_LR
)

print("\nLinear Regression Results:")

print("Mean Absolute Error:", f"{mae_LR:.2f}")

print("Mean Absolute Percentage Error:", f"{mape_LR * 100:.2f}%")

print("R2 Score:", f"{r2_LR:.4f}")


# ============================================================
# 23. Compare Model Performance
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Support Vector Regression",
        "Random Forest Regression",
        "Linear Regression"
    ],

    "MAE": [
        mae_SVR,
        mae_RFR,
        mae_LR
    ],

    "MAPE (%)": [
        mape_SVR * 100,
        mape_RFR * 100,
        mape_LR * 100
    ],

    "R2 Score": [
        r2_SVR,
        r2_RFR,
        r2_LR
    ]

})

print("\n======================================")

print("MODEL PERFORMANCE COMPARISON")

print("======================================")

print(results.to_string(index=False))


# ============================================================
# 24. Visualize Model Performance
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results,
    x="Model",
    y="MAPE (%)"
)

plt.title("Model Comparison - Mean Absolute Percentage Error")

plt.xlabel("Machine Learning Models")

plt.ylabel("MAPE (%)")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()


# ============================================================
# 25. Visualize Actual vs Predicted House Prices
# ============================================================

predictions = {
    "SVR": Y_pred_SVR,
    "Random Forest": Y_pred_RFR,
    "Linear Regression": Y_pred_LR
}

for model_name, predictions_values in predictions.items():

    plt.figure(figsize=(8, 6))

    plt.scatter(
        Y_valid,
        predictions_values,
        alpha=0.6
    )

    # Reference line
    min_value = min(
        Y_valid.min(),
        predictions_values.min()
    )

    max_value = max(
        Y_valid.max(),
        predictions_values.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.title(
        f"Actual vs Predicted House Prices - {model_name}"
    )

    plt.xlabel("Actual House Prices")

    plt.ylabel("Predicted House Prices")

    plt.tight_layout()

    plt.show()


# ============================================================
# 26. Final Results
# ============================================================

best_model = results.loc[
    results["MAPE (%)"].idxmin()
]

print("\n======================================")

print("HOUSE PRICE PREDICTION RESULTS")

print("======================================")

print("Model with Lowest Validation MAPE:")

print(best_model["Model"])

print(
    "Validation MAPE:",
    f"{best_model['MAPE (%)']:.2f}%"
)

print(
    "Validation MAE:",
    f"{best_model['MAE']:.2f}"
)

print(
    "Validation R2 Score:",
    f"{best_model['R2 Score']:.4f}"
)

print("\nHouse Price Prediction Completed Successfully!")

