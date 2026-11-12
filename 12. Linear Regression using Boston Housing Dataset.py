
# ============================================================
# BOSTON HOUSE PRICE PREDICTION USING LINEAR REGRESSION
# ============================================================


# 1. Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ============================================================
# 2. Load the Boston Housing Dataset
# ============================================================

print("\nLoading Boston Housing Dataset...")

try:
    boston = fetch_openml(
        name="boston",
        version=1,
        as_frame=True
    )

except Exception as error:
    print("Error loading dataset:", error)
    print("Check your internet connection and try again.")
    raise SystemExit


# ============================================================
# 3. Explore the Dataset
# ============================================================

# Display dataset dimensions

print("\nDataset Shape:")
print(boston.data.shape)

# Display feature names

print("\nFeature Names:")
print(boston.feature_names)


# ============================================================
# 4. Create a DataFrame
# ============================================================

data = pd.DataFrame(
    boston.data,
    columns=boston.feature_names
)

# Add target variable (House Price)

data["Price"] = pd.to_numeric(
    boston.target,
    errors="coerce"
)

# Display first 10 rows

print("\nFirst 10 Rows:")
print(data.head(10))


# ============================================================
# 5. Understand the Dataset
# ============================================================

# Statistical summary

print("\nDataset Description:")
print(data.describe())

# Dataset information

print("\nDataset Information:")
data.info()

# Check missing values

print("\nMissing Values:")
print(data.isnull().sum())



# ============================================================
# 6. Prepare Features and Target
# ============================================================

# Independent variables

X = data.drop("Price", axis=1)

# Convert all feature columns into numeric values

X = X.apply(pd.to_numeric, errors="coerce")

# Dependent variable

y = pd.to_numeric(data["Price"], errors="coerce")

# Combine features and target

combined_data = pd.concat([X, y.rename("Price")], axis=1)

# Remove rows containing missing values

combined_data = combined_data.dropna()

# Separate features and target again

X = combined_data.drop("Price", axis=1)
y = combined_data["Price"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Verify that all features are numeric

print("\nFeature Data Types:")
print(X.dtypes)

print("\nAre all features numeric?")
print(all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes))


# ============================================================
# 7. Split the Dataset into Training and Testing Sets
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

print("\nTraining and Testing Data Shapes:")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 8. Create and Train the Linear Regression Model
# ============================================================

regressor = LinearRegression()

regressor.fit(X_train, y_train)

print("\nLinear Regression Model Training Completed!")


print("\nChecking Test Data Types:")
print(X_test.dtypes)

print("\nChecking for non-numeric values:")
print(X_test.select_dtypes(exclude=[np.number]).columns.tolist())


# ============================================================
# 9. Make Predictions
# ============================================================

y_pred = regressor.predict(X_test)

print("\nActual and Predicted House Prices:")

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(comparison.head(10))


# ============================================================
# 10. Visualize Actual vs Predicted House Prices
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    color="green",
    alpha=0.7
)

# Reference line showing perfect predictions

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Price: in $1000's")
plt.ylabel("Predicted Price: in $1000's")

plt.title(
    "Actual vs Predicted House Prices: Linear Regression"
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 11. Evaluate Model Performance
# ============================================================

# Mean Squared Error

mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error

rmse = np.sqrt(mse)

# Mean Absolute Error

mae = mean_absolute_error(y_test, y_pred)

# R-squared Score

r2 = r2_score(y_test, y_pred)


# Display evaluation results

print("\nModel Performance:")

print("Mean Squared Error:", round(mse, 4))

print("Root Mean Squared Error:", round(rmse, 4))

print("Mean Absolute Error:", round(mae, 4))

print("R-squared Score:", round(r2, 4))


# ============================================================
# 12. Visualize Model Performance
# ============================================================

metrics = ["MSE", "RMSE", "MAE"]

values = [mse, rmse, mae]

plt.figure(figsize=(8, 5))

plt.bar(metrics, values)

plt.title("Linear Regression Model Error Metrics")

plt.xlabel("Evaluation Metrics")
plt.ylabel("Error Value")

plt.tight_layout()
plt.show()


# ============================================================
# 13. Display Model Coefficients
# ============================================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": regressor.coef_
})

print("\nLinear Regression Coefficients:")

print(coefficients)

print("\nIntercept:", regressor.intercept_)


# ============================================================
# 14. Final Result
# ============================================================

print("\n======================================")
print("BOSTON HOUSE PRICE PREDICTION COMPLETED")
print("======================================")

print("Model: Linear Regression")
print("Number of Training Samples:", len(X_train))
print("Number of Testing Samples:", len(X_test))
print("R-squared Score:", round(r2, 4))
