# ============================================================
# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# ============================================================

# STEP 1: IMPORT LIBRARIES

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# ============================================================
# STEP 2: LOAD THE DATASET
# ============================================================

df = pd.read_csv("new.csv")

if df.empty:
    raise ValueError("The dataset is empty.")

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe().T)

print("\nUnique values in each column:")
print(df.nunique())


# ============================================================
# STEP 3: CHECK AND HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")

for col in df.columns:
    missing_count = df[col].isnull().sum()

    if missing_count > 0:
        print(
            f"Column '{col}' contains "
            f"{missing_count} missing values."
        )

# Remove rows containing missing values
df = df.dropna().copy()

print(
    "\nTotal records after removing missing values:",
    len(df)
)

if df.empty:
    raise ValueError(
        "No records remain after removing missing values."
    )


# ============================================================
# STEP 4: PROCESS THE CUSTOMER REGISTRATION DATE
# ============================================================

if "Dt_Customer" in df.columns:

    # The common marketing dataset stores dates as day-month-year.
    # dayfirst=True handles dates such as 04-09-2012 correctly.
    parsed_dates = pd.to_datetime(
        df["Dt_Customer"],
        dayfirst=True,
        errors="coerce"
    )

    invalid_dates = parsed_dates.isna().sum()

    if invalid_dates > 0:
        print(
            f"\nRemoving {invalid_dates} records "
            "with invalid customer registration dates."
        )

        df = df.loc[parsed_dates.notna()].copy()
        parsed_dates = parsed_dates.loc[df.index]

    if df.empty:
        raise ValueError(
            "No valid records remain after parsing dates."
        )

    df["day"] = parsed_dates.dt.day
    df["month"] = parsed_dates.dt.month
    df["year"] = parsed_dates.dt.year

    # Remove the original date column
    df.drop(columns=["Dt_Customer"], inplace=True)

# ============================================================
# STEP 5: REMOVE CONSTANT OR UNINFORMATIVE COLUMNS
# ============================================================

# These columns are constant in the standard marketing dataset.
# Drop them only if they exist.
columns_to_drop = [
    "Z_CostContact",
    "Z_Revenue"
]

df.drop(
    columns=columns_to_drop,
    errors="ignore",
    inplace=True
)

print("\nDataset shape after date processing:")
print(df.shape)


# ============================================================
# STEP 6: IDENTIFY CATEGORICAL AND NUMERIC COLUMNS
# ============================================================

objects = df.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nCategorical columns:")
print(objects)

print("\nNumeric columns:")
print(numeric_columns)

# ============================================================
# STEP 7: VISUALIZE CATEGORICAL FEATURES
# ============================================================

if objects:

    n_cols = 2
    n_rows = int(np.ceil(len(objects) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 5 * n_rows),
        squeeze=False
    )

    for i, col in enumerate(objects):

        ax = axes.flatten()[i]

        order = df[col].value_counts().index

        sb.countplot(
            data=df,
            x=col,
            order=order,
            ax=ax
        )

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Customer Count")
        ax.tick_params(axis="x", rotation=35)

    # Hide unused axes
    for j in range(len(objects), len(axes.flatten())):
        axes.flatten()[j].set_visible(False)

    plt.tight_layout()
    plt.show()

else:
    print("\nNo categorical columns are available to plot.")


# ============================================================
# STEP 8: ANALYZE MARITAL STATUS
# ============================================================

if "Marital_Status" in df.columns:

    print("\nMarital status distribution:")
    print(df["Marital_Status"].value_counts())

    plt.figure(figsize=(9, 5))

    sb.countplot(
        data=df,
        x="Marital_Status",
        order=df["Marital_Status"].value_counts().index
    )

    plt.title("Customer Distribution by Marital Status")
    plt.xlabel("Marital Status")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 9: VISUALIZE CATEGORICAL FEATURES BY RESPONSE
# ============================================================

if "Response" in df.columns and objects:

    # Convert response to categorical labels for plotting.
    # This does not modify the original Response column.
    response_plot = df["Response"].astype(str)

    n_cols = 2
    n_rows = int(np.ceil(len(objects) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 5 * n_rows),
        squeeze=False
    )

    for i, col in enumerate(objects):

        ax = axes.flatten()[i]

        plot_df = df[[col]].copy()
        plot_df["Response_Group"] = response_plot

        sb.countplot(
            data=plot_df,
            x=col,
            hue="Response_Group",
            ax=ax
        )

        ax.set_title(f"{col} by Response")
        ax.set_xlabel(col)
        ax.set_ylabel("Customer Count")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Response")

    for j in range(len(objects), len(axes.flatten())):
        axes.flatten()[j].set_visible(False)

    plt.tight_layout()
    plt.show()

else:
    print(
        "\nSkipping response analysis: "
        "Response or categorical features are unavailable."
    )


# ============================================================
# STEP 10: ENCODE CATEGORICAL VARIABLES
# ============================================================

# Keep the encoders for reference.
label_encoders = {}

for col in objects:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

    label_encoders[col] = encoder

print("\nDataset after categorical encoding:")
print(df.head())

# ============================================================
# STEP 11: CORRELATION ANALYSIS
# ============================================================

numeric_df = df.select_dtypes(
    include=[np.number]
)

correlation = numeric_df.corr()

plt.figure(figsize=(15, 12))

sb.heatmap(
    correlation > 0.8,
    annot=True,
    fmt="d",
    cmap="coolwarm",
    cbar=False
)

plt.title("Feature Correlation Greater Than 0.8")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: PREPARE FEATURES FOR CLUSTERING
# ============================================================

# Exclude ID because it is an identifier, not a customer behavior
# feature. Retain Response to follow the original workflow if
# the column exists; for unsupervised segmentation, you may wish
# to exclude Response as well.

excluded_columns = ["ID"]

# Exclude the response outcome from segmentation to avoid using
# a campaign outcome to define customer groups.
excluded_columns.append("Response")

feature_df = df.drop(
    columns=excluded_columns,
    errors="ignore"
).copy()

# Keep numeric columns only
feature_df = feature_df.select_dtypes(
    include=[np.number]
)

# Replace infinite values, then remove any affected rows.
feature_df = feature_df.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_rows = feature_df.notna().all(axis=1)

feature_df = feature_df.loc[valid_rows].copy()
df = df.loc[valid_rows].copy()

if feature_df.empty:
    raise ValueError(
        "No usable numeric features remain for clustering."
    )

if feature_df.shape[1] == 0:
    raise ValueError(
        "No numeric feature columns are available for clustering."
    )

if len(feature_df) < 3:
    raise ValueError(
        "At least three valid customer records are required "
        "for t-SNE visualization."
    )

print("\nFeatures used for clustering:")
print(feature_df.columns.tolist())

print("\nClustering feature matrix shape:")
print(feature_df.shape)


# ============================================================
# STEP 13: STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(feature_df)

# Use a DataFrame for readable downstream operations
scaled_df = pd.DataFrame(
    scaled_data,
    columns=feature_df.columns,
    index=feature_df.index
)

print("\nStandardized feature matrix:")
print(scaled_df.head())

# ============================================================
# STEP 14: VISUALIZE CUSTOMER DATA USING t-SNE
# ============================================================

# t-SNE perplexity must be smaller than the number of samples.
# A small dataset therefore needs a smaller perplexity.
perplexity = min(30, max(1, (len(scaled_df) - 1) // 3))

tsne_model = TSNE(
    n_components=2,
    random_state=0,
    perplexity=perplexity,
    init="pca",
    learning_rate="auto"
)

tsne_data = tsne_model.fit_transform(scaled_data)

df_tsne = pd.DataFrame(
    {
        "x": tsne_data[:, 0],
        "y": tsne_data[:, 1]
    },
    index=scaled_df.index
)

plt.figure(figsize=(8, 7))

sb.scatterplot(
    data=df_tsne,
    x="x",
    y="y",
    s=40
)

plt.title("Customer Visualization Using t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 15: ELBOW METHOD TO SELECT NUMBER OF CLUSTERS
# ============================================================

# Limit the maximum number of clusters to the number of samples.
max_clusters = min(20, len(scaled_df))

cluster_range = range(1, max_clusters + 1)

error = []

for n_clusters in cluster_range:

    kmeans = KMeans(
        init="k-means++",
        n_clusters=n_clusters,
        max_iter=500,
        random_state=22,
        n_init=10
    )

    kmeans.fit(scaled_data)

    error.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))

sb.lineplot(
    x=list(cluster_range),
    y=error,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.xticks(list(cluster_range))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# STEP 16: APPLY K-MEANS CLUSTERING
# ============================================================

# Use five clusters, as in the original code, provided that
# there are at least five valid customer records.
n_clusters = min(5, len(scaled_df))

kmeans_model = KMeans(
    init="k-means++",
    n_clusters=n_clusters,
    max_iter=500,
    random_state=22,
    n_init=10
)

segments = kmeans_model.fit_predict(scaled_data)

# Store the assigned segment for each customer.
df["Segment"] = -1
df.loc[scaled_df.index, "Segment"] = segments

df["Segment"] = df["Segment"].astype(int)

print("\nCustomer segment counts:")
print(df["Segment"].value_counts().sort_index())


# ============================================================
# STEP 17: VISUALIZE CUSTOMER SEGMENTS
# ============================================================

df_tsne["segment"] = segments.astype(str)

plt.figure(figsize=(9, 7))

sb.scatterplot(
    data=df_tsne,
    x="x",
    y="y",
    hue="segment",
    palette="tab10",
    s=50
)

plt.title("Customer Segments Visualized Using t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Customer Segment")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 18: PROFILE THE CUSTOMER SEGMENTS
# ============================================================

# Calculate mean feature values for each segment using the
# original, unscaled feature values.
segment_profile = feature_df.copy()
segment_profile["Segment"] = segments

segment_summary = segment_profile.groupby(
    "Segment"
).mean().round(2)

print("\nAverage feature values for each customer segment:")
print(segment_summary)

print("\nNumber of customers in each segment:")
print(
    df["Segment"]
    .value_counts()
    .sort_index()
    .rename("Customer Count")
    .to_frame()
)

# ============================================================
# STEP 19: SAVE THE RESULTS
# ============================================================

df.to_csv(
    "customer_segments.csv",
    index=False
)

segment_summary.to_csv(
    "customer_segment_summary.csv"
)

print("\nProject completed successfully.")
print("Saved customer assignments to: customer_segments.csv")
print("Saved segment profiles to: customer_segment_summary.csv")




print('Done')
