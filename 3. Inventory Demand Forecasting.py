
# ============================================================
# INVENTORY DEMAND FORECASTING USING MACHINE LEARNING
# ============================================================

# STEP 1: IMPORT LIBRARIES

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import holidays

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor


# ============================================================
# STEP 2: LOAD THE DATASET
# ============================================================

df = pd.read_csv("train.csv")

# Validate the required columns
required_columns = {"date", "store", "item", "sales"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

print("First five rows:")
print(df.head())

print("Last five rows:")
print(df.tail())

print("Dataset shape:", df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# STEP 3: DATA CLEANING AND DATE CONVERSION
# ============================================================

# Convert dates safely
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# Convert numeric columns safely
for col in ["store", "item", "sales"]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Remove rows with missing essential values
df = df.dropna(
    subset=["date", "store", "item", "sales"]
).copy()

# Remove duplicate records
df = df.drop_duplicates().copy()

# Remove negative sales values
df = df[df["sales"] >= 0].copy()

# Extract date components
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

print("\nDataset after cleaning:")
print(df.shape)

print(df.head())


# ============================================================
# STEP 4: CREATE WEEKEND AND WEEKDAY FEATURES
# ============================================================

# Monday = 0, Sunday = 6
df["weekday"] = df["date"].dt.weekday

# Saturday and Sunday = 1; weekdays = 0
df["weekend"] = (
    df["weekday"] >= 5
).astype(int)


# ============================================================
# STEP 5: CREATE PUBLIC HOLIDAY FEATURE
# ============================================================

# Indian public holiday calendar
india_holidays = holidays.country_holidays("IN")

# Convert Timestamp to date before checking holiday membership
df["holidays"] = df["date"].apply(
    lambda x: int(x.date() in india_holidays)
)


# ============================================================
# STEP 6: CYCLICAL MONTH ENCODING
# ============================================================

df["m1"] = np.sin(
    df["month"] * (2 * np.pi / 12)
)

df["m2"] = np.cos(
    df["month"] * (2 * np.pi / 12)
)

print(df.head())


# ============================================================
# STEP 7: EXPLORE STORES AND PRODUCTS
# ============================================================

print("Number of unique stores:", df["store"].nunique())
print("Number of unique products:", df["item"].nunique())


# ============================================================
# STEP 8: VISUALIZE SALES BY FEATURE
# ============================================================

features = [
    "store",
    "year",
    "month",
    "weekday",
    "weekend",
    "holidays"
]

fig, axes = plt.subplots(
    2, 3,
    figsize=(20, 10)
)

for ax, col in zip(axes.flatten(), features):

    average_sales = (
        df.groupby(col)["sales"]
        .mean()
        .sort_index()
    )

    average_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(f"Average Sales by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Average Sales")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 9: VISUALIZE SALES BY DAY OF THE MONTH
# ============================================================

plt.figure(figsize=(10, 5))

df.groupby("day")["sales"].mean().plot()

plt.title("Average Sales by Day of the Month")
plt.xlabel("Day")
plt.ylabel("Average Sales")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 10: 30-DAY SIMPLE MOVING AVERAGE
# ============================================================

# Aggregate sales by date to create a genuine daily time series
data = (
    df.loc[df["year"] == 2013]
    .groupby("date")["sales"]
    .sum()
    .sort_index()
)

if len(data) >= 30:

    window_size = 30

    sma = data.rolling(
        window=window_size
    ).mean()

    plt.figure(figsize=(15, 6))

    data.plot(label="Daily Total Sales")

    sma.plot(
        label="30-Day Simple Moving Average",
        linewidth=2
    )

    plt.title("Daily Sales and 30-Day Moving Average (2013)")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

else:
    print(
        "Not enough daily observations in 2013 "
        "to calculate a 30-day moving average."
    )


# ============================================================
# STEP 11: SALES DISTRIBUTION AND OUTLIERS
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5)
)

sb.histplot(
    data=df["sales"],
    kde=True,
    ax=axes[0]
)

axes[0].set_title("Sales Distribution")
axes[0].set_xlabel("Sales")

sb.boxplot(
    x=df["sales"],
    ax=axes[1]
)

axes[1].set_title("Sales Boxplot")
axes[1].set_xlabel("Sales")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: CORRELATION ANALYSIS
# ============================================================

numeric_df = df.select_dtypes(
    include=[np.number]
)

correlation = numeric_df.corr()

plt.figure(figsize=(12, 10))

sb.heatmap(
    correlation > 0.8,
    annot=True,
    fmt="d",
    cmap="coolwarm",
    cbar=False
)

plt.title("Features with Correlation Greater Than 0.8")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 13: FILTER EXTREME SALES VALUES
# ============================================================

# Following the original case study's threshold
df = df[df["sales"] < 140].copy()

print("Shape after filtering sales below 140:", df.shape)

if df.empty:
    raise ValueError(
        "No records remain after filtering. "
        "Check the sales threshold and dataset."
    )


# ============================================================
# STEP 14: PREPARE FEATURES AND TARGET
# ============================================================

# Remove date, target, and year from the predictors
features = df.drop(
    columns=["date", "sales", "year"]
)

target = df["sales"].to_numpy()

# Ensure all predictors are numeric and finite
if not all(
    pd.api.types.is_numeric_dtype(features[col])
    for col in features.columns
):
    raise ValueError(
        "All model features must be numeric."
    )

if not np.isfinite(features.to_numpy(dtype=float)).all():
    raise ValueError(
        "Features contain missing or infinite values."
    )

if not np.isfinite(target).all():
    raise ValueError(
        "The target contains missing or infinite values."
    )

print("\nFeature columns:")
print(features.columns.tolist())

print("Feature shape:", features.shape)
print("Target shape:", target.shape)


# ============================================================
# STEP 15: SPLIT TRAINING AND VALIDATION DATA
# ============================================================

X_train, X_val, Y_train, Y_val = train_test_split(
    features,
    target,
    test_size=0.05,
    random_state=22
)

print("Training feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)


# ============================================================
# STEP 16: SCALE THE FEATURES
# ============================================================

# Fit the scaler on training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# ============================================================
# STEP 17: INITIALIZE REGRESSION MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "XGBoost Regressor": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=22,
        n_jobs=-1
    ),

    "Lasso Regression": Lasso(
        alpha=0.01,
        max_iter=10000
    ),

    "Ridge Regression": Ridge(
        alpha=1.0
    )
}


# ============================================================
# STEP 18: TRAIN AND EVALUATE THE MODELS
# ============================================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\n{'=' * 50}")
    print(f"Model: {name}")
    print("=" * 50)

    # Train model
    model.fit(
        X_train_scaled,
        Y_train
    )

    # Predict training and validation sales
    train_preds = model.predict(X_train_scaled)
    val_preds = model.predict(X_val_scaled)

    # Calculate MAE
    train_error = mae(
        Y_train,
        train_preds
    )

    val_error = mae(
        Y_val,
        val_preds
    )

    print(f"Training MAE:   {train_error:.4f}")
    print(f"Validation MAE: {val_error:.4f}")

    # Save model results
    results.append({
        "Model": name,
        "Training MAE": train_error,
        "Validation MAE": val_error
    })

    trained_models[name] = model


# ============================================================
# STEP 19: COMPARE MODEL PERFORMANCE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Validation MAE",
    ascending=True
).reset_index(drop=True)

print("\nFinal Model Comparison:")
print(results_df)

best_model_name = results_df.loc[0, "Model"]
best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)
print(
    "Lowest validation MAE:",
    round(results_df.loc[0, "Validation MAE"], 4)
)


# ============================================================
# STEP 20: PLOT VALIDATION ERRORS
# ============================================================

plt.figure(figsize=(10, 5))

sb.barplot(
    data=results_df,
    x="Model",
    y="Validation MAE"
)

plt.title("Validation MAE by Model")
plt.xlabel("Machine Learning Model")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 21: ACTUAL VS PREDICTED SALES
# ============================================================

val_preds = best_model.predict(X_val_scaled)

plt.figure(figsize=(8, 8))

plt.scatter(
    Y_val,
    val_preds,
    alpha=0.3
)

min_value = min(Y_val.min(), val_preds.min())
max_value = max(Y_val.max(), val_preds.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title(f"Actual vs Predicted Sales: {best_model_name}")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 22: FINAL SUMMARY
# ============================================================

print("\n========== PROJECT SUMMARY ==========")
print("Total records:", len(df))
print("Number of stores:", df["store"].nunique())
print("Number of products:", df["item"].nunique())
print("Number of models evaluated:", len(models))
print("Best model:", best_model_name)
print(
    "Best validation MAE:",
    round(results_df.loc[0, "Validation MAE"], 4)
)
print("=====================================")
