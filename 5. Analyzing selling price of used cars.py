# ============================================================
# Automobile Data Analysis and Visualization
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------

df = pd.read_csv("output.csv")

# Remove the first column only if it is an exported index column
if (
    len(df.columns) > 0
    and (
        str(df.columns[0]).lower().startswith("unnamed")
        or str(df.columns[0]).lower() in ["index", "row number"]
    )
):
    df = df.iloc[:, 1:]

# Expected automobile dataset columns
headers = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price"
]

# Validate the dataset structure before renaming columns
if len(df.columns) != len(headers):
    raise ValueError(
        f"Expected {len(headers)} columns, but found "
        f"{len(df.columns)}. Check output.csv and whether "
        "an extra index column needs to be removed."
    )

df.columns = headers

# Display the first five rows
print("\nFirst five rows:")
print(df.head())


# ------------------------------------------------------------
# 2. Inspect the dataset
# ------------------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all").T)

print("\nMissing values in each column:")
print(df.isna().sum())

print("\nColumns containing missing values:")
print(df.isna().any())

# ------------------------------------------------------------
# 3. Handle missing values and invalid entries
# ------------------------------------------------------------

# Treat question marks as missing values
df.replace("?", np.nan, inplace=True)

# Convert numeric columns to numeric data types
numeric_columns = [
    "symboling",
    "normalized-losses",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-size",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove records without a valid price because price is
# required for the analysis and visualizations
df.dropna(subset=["price"], inplace=True)

# Price must be numeric
df["price"] = df["price"].astype(float)

# Handle missing numerical values using column medians.
# Columns that are completely missing are left unchanged.
for col in numeric_columns:
    if col != "price" and df[col].notna().any():
        df[col] = df[col].fillna(df[col].median())

# Remove rows with missing fuel-efficiency values before
# performing the reciprocal conversion
df.loc[df["city-mpg"] <= 0, "city-mpg"] = np.nan

# ------------------------------------------------------------
# 4. Convert city fuel efficiency
# ------------------------------------------------------------

# Miles per gallon to litres per 100 kilometres
df["city-L / 100km"] = 235.215 / df["city-mpg"]

# Remove the original city-mpg column after conversion
df.drop(columns=["city-mpg"], inplace=True)

print("\nUpdated column names:")
print(df.columns.tolist())

print("\nUpdated data types:")
print(df.dtypes)

print("\nUnique price values:")
print(df["price"].unique())

# ------------------------------------------------------------
# 5. Normalize length, width and height
# ------------------------------------------------------------

for col in ["length", "width", "height"]:
    max_value = df[col].max()

    if pd.notna(max_value) and max_value != 0:
        df[col] = df[col] / max_value

print("\nNormalized length, width and height:")
print(df[["length", "width", "height"]].head())


# ------------------------------------------------------------
# 6. Binning: Group prices into categories
# ------------------------------------------------------------

if df["price"].nunique() >= 3:
    bins = np.linspace(
        df["price"].min(),
        df["price"].max(),
        4
    )

    group_names = ["Low", "Medium", "High"]

    df["price-binned"] = pd.cut(
        df["price"],
        bins=bins,
        labels=group_names,
        include_lowest=True,
        duplicates="drop"
    )
else:
    df["price-binned"] = pd.cut(
        df["price"],
        bins=3,
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

print("\nBinned prices:")
print(df["price-binned"].value_counts(sort=False))

plt.figure(figsize=(8, 5))
df["price-binned"].value_counts(sort=False).plot(
    kind="bar",
    color="steelblue",
    edgecolor="black"
)
plt.title("Distribution of Cars by Price Category")
plt.xlabel("Price Category")
plt.ylabel("Number of Cars")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Convert categorical variables into dummy variables
# ------------------------------------------------------------

fuel_type_dummies = pd.get_dummies(
    df["fuel-type"],
    prefix="fuel",
    dtype=int
)

print("\nFuel-type dummy variables:")
print(fuel_type_dummies.head())


# ------------------------------------------------------------
# 8. Descriptive statistics
# ------------------------------------------------------------

print("\nNumerical descriptive statistics:")
print(df.describe())

# ------------------------------------------------------------
# 9. Boxplot of price
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.boxplot(df["price"].dropna())
plt.title("Boxplot of Automobile Prices")
plt.ylabel("Price")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. Boxplot: Drive wheels vs. price
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))
sns.boxplot(
    x="drive-wheels",
    y="price",
    data=df
)
plt.title("Drive Wheels vs. Price")
plt.xlabel("Drive Wheels")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 11. Scatterplot: Engine size vs. price
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(
    df["engine-size"],
    df["price"],
    alpha=0.7
)
plt.title("Scatterplot of Engine Size vs. Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. Group data by drive wheels and body style
# ------------------------------------------------------------

test = df[["drive-wheels", "body-style", "price"]].copy()

data_grp = test.groupby(
    ["drive-wheels", "body-style"],
    as_index=False,
    observed=True
)["price"].mean()

print("\nAverage price by drive wheels and body style:")
print(data_grp)


# ------------------------------------------------------------
# 13. Create a pivot table
# ------------------------------------------------------------

data_pivot = data_grp.pivot(
    index="drive-wheels",
    columns="body-style",
    values="price"
)

print("\nPivot table:")
print(data_pivot)

# Visualize the pivot table using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    data_pivot,
    annot=True,
    fmt=".0f",
    cmap="RdBu_r",
    linewidths=0.5,
    mask=data_pivot.isna()
)
plt.title("Average Price by Drive Wheels and Body Style")
plt.xlabel("Body Style")
plt.ylabel("Drive Wheels")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 14. ANOVA: Compare Honda and Subaru prices
# ------------------------------------------------------------

data_annova = df[["make", "price"]].copy()

honda_prices = data_annova.loc[
    data_annova["make"].str.lower() == "honda",
    "price"
].dropna()

subaru_prices = data_annova.loc[
    data_annova["make"].str.lower() == "subaru",
    "price"
].dropna()

print("\nANOVA: Honda vs. Subaru")

if len(honda_prices) >= 2 and len(subaru_prices) >= 2:
    annova_results = sp.stats.f_oneway(
        honda_prices,
        subaru_prices
    )

    print("F-statistic:", annova_results.statistic)
    print("P-value:", annova_results.pvalue)

    alpha = 0.05

    if annova_results.pvalue < alpha:
        print(
            "Conclusion: There is a statistically significant "
            "difference between the mean prices of Honda and "
            "Subaru vehicles at the 5% significance level."
        )
    else:
        print(
            "Conclusion: There is insufficient evidence of a "
            "difference between the mean prices of Honda and "
            "Subaru vehicles at the 5% significance level."
        )
else:
    print(
        "ANOVA cannot be performed reliably because at least "
        "one manufacturer has fewer than two valid price values."
    )

# ------------------------------------------------------------
# 15. Regression plot: Engine size vs. price
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.regplot(
    x="engine-size",
    y="price",
    data=df,
    scatter_kws={"alpha": 0.6}
)
plt.title("Regression Plot: Engine Size vs. Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.ylim(bottom=0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 16. Save the cleaned dataset
# ------------------------------------------------------------

df.to_csv("cleaned_automobile_data.csv", index=False)

print(
    "\nAnalysis completed successfully. "
    "Cleaned data saved as cleaned_automobile_data.csv."
)

print('Done')
