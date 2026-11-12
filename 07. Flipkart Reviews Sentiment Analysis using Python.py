
# ============================================================
# FLIPKART REVIEWS SENTIMENT ANALYSIS
# ============================================================

# 1. Import Required Libraries

import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud


# ============================================================
# 2. Load the Dataset
# ============================================================

# Get the folder where this Python file is saved
BASE_DIR = Path(__file__).resolve().parent

# Dataset file path
file_path = BASE_DIR / "flipkart_data.csv"

# Check whether the dataset exists
if not file_path.exists():
    raise FileNotFoundError(
        "flipkart_data.csv was not found. "
        "Place it in the same folder as this Python file."
    )

# Read the CSV file
try:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="latin-1")

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

# ============================================================
# 3. Check and Clean the Dataset
# ============================================================

# Check required columns
required_columns = ["review", "rating"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing from the dataset."
        )

# Remove rows with missing reviews or ratings
df = df.dropna(subset=["review", "rating"]).copy()

# Convert ratings into numeric values
df["rating"] = pd.to_numeric(
    df["rating"], errors="coerce"
)

# Remove invalid ratings
df = df.dropna(subset=["rating"]).copy()

# Convert reviews into strings and remove extra spaces
df["review"] = df["review"].astype(str).str.strip()

# Remove empty reviews
df = df[df["review"] != ""].copy()

if df.empty:
    raise ValueError("No valid reviews found in the dataset.")

print("\nDataset cleaned successfully!")
print("Cleaned Dataset Shape:", df.shape)


# ============================================================
# 4. Download NLTK Stopwords
# ============================================================

# Download English stopwords
nltk.download("stopwords", quiet=True)

# Load English stopwords
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    raise RuntimeError(
        "NLTK stopwords could not be downloaded. "
        "Check your internet connection and try again."
    )

print("\nEnglish stopwords loaded successfully!")


# ============================================================
# 5. Preprocess Customer Reviews
# ============================================================

def preprocess_reviews_stopwords(data):

    # Create a copy of the original dataset
    data = data.copy()

    # Convert reviews into lowercase
    data["review"] = data["review"].str.lower()

    # Remove English stopwords
    data["review"] = data["review"].apply(
        lambda x: " ".join(
            word for word in x.split()
            if word not in stop_words
        )
    )

    # Create sentiment labels
    # Rating >= 4 = Positive (1)
    # Rating < 4 = Negative (0)

    data["sentiment"] = data["rating"].apply(
        lambda x: 1 if x >= 4 else 0
    )

    # Remove reviews that became empty
    data = data[data["review"].str.strip() != ""].copy()

    return data


# Apply preprocessing
df_cleaned = preprocess_reviews_stopwords(df)

print("\nPreprocessing Completed!")

print("\nCleaned Reviews:")
print(df_cleaned.head())

# Check whether both sentiment classes exist
if df_cleaned["sentiment"].nunique() < 2:
    raise ValueError(
        "The dataset must contain both positive and negative reviews."
    )


# ============================================================
# 6. Visualize Sentiment Distribution
# ============================================================

# Count positive and negative reviews
sentiment_counts = df_cleaned["sentiment"].value_counts()

# Ensure the order is Negative (0), Positive (1)
sentiment_counts = sentiment_counts.reindex(
    [0, 1], fill_value=0
)

# Create bar chart
plt.figure(figsize=(6, 4))

sentiment_counts.plot(
    kind="bar",
    color=["red", "green"]
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.xticks(
    ticks=[0, 1],
    labels=["Negative", "Positive"],
    rotation=0
)

plt.tight_layout()
plt.show()


# ============================================================
# 7. Generate Word Cloud for Positive Reviews
# ============================================================

# Select positive reviews
positive_reviews = df_cleaned[
    df_cleaned["sentiment"] == 1
]["review"]

# Combine all positive reviews
positive_text = " ".join(positive_reviews.tolist()).strip()

# Generate word cloud
if positive_text:

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(positive_text)

    plt.figure(figsize=(8, 6))

    plt.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    plt.axis("off")

    plt.title("Word Cloud for Positive Reviews")

    plt.tight_layout()
    plt.show()

else:
    print("No positive reviews available for Word Cloud.")


# ============================================================
# 8. Separate Features and Target
# ============================================================

# Input feature: Customer review text
X = df_cleaned["review"]

# Target variable: Sentiment
y = df_cleaned["sentiment"]


# ============================================================
# 9. Split Training and Testing Data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.value_counts().min() >= 2 else None
)

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))


# ============================================================
# 10. Convert Text into Numerical Features Using TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=5000
)

# Fit vectorizer using training data only
X_train = vectorizer.fit_transform(X_train)

# Transform testing data using the same vectorizer
X_test = vectorizer.transform(X_test)

print("\nTF-IDF Vectorization Completed!")

print("Training Feature Shape:", X_train.shape)
print("Testing Feature Shape:", X_test.shape)


# ============================================================
# 11. Train Decision Tree Classifier
# ============================================================

model = DecisionTreeClassifier(
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nDecision Tree Model Training Completed!")


# ============================================================
# 12. Predict Customer Sentiment
# ============================================================

y_pred = model.predict(X_test)

print("\nPredictions Completed!")


# ============================================================
# 13. Evaluate Model Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# 14. Generate Confusion Matrix
# ============================================================

conf_matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

plt.figure(figsize=(6, 4))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")

plt.tight_layout()
plt.show()


# ============================================================
# 9. Split Training and Testing Data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.value_counts().min() >= 2 else None
)

# Preserve original test reviews
X_test_reviews = X_test.copy()

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

