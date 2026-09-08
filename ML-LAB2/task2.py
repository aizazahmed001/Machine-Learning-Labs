# Lab Task: Basic EDA on a chosen dataset
# Dataset: Diamonds (diamonds.csv)
# Context: Diamond characteristics (carat, cut, color, clarity, dimensions)
#          and their market price - a classic pricing/valuation dataset.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("diamonds.csv")

# ---------------------------------------------------------
# Step 2: Overview of the dataset
# ---------------------------------------------------------
print("Shape (rows, columns):", df.shape)
print("\nColumn info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

# Identify variable types
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()
print("\nNumeric variables:", numeric_cols)
print("Categorical variables:", categorical_cols)

# ---------------------------------------------------------
# Step 3: Summary statistics for key numeric variables
# ---------------------------------------------------------
print("\nSummary statistics (numeric columns):")
summary = df[numeric_cols].describe().T
summary["median"] = df[numeric_cols].median()
print(summary[["mean", "median", "std", "min", "max"]])

# Frequency counts for categorical variables
for col in categorical_cols:
    print(f"\nValue counts for '{col}':")
    print(df[col].value_counts())

# ---------------------------------------------------------
# Step 4: Visualizations
# ---------------------------------------------------------

# 4a. Histograms of all numeric features
df[numeric_cols].hist(figsize=(15, 10), bins=30)
plt.suptitle("Distribution of Numeric Features")
plt.tight_layout()
plt.savefig("hist_numeric_features.png", dpi=150)
plt.show()

# 4b. Boxplots to compare price across categorical variables
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axes, ["cut", "color", "clarity"]):
    sns.boxplot(x=df[col], y=df["price"], ax=ax)
    ax.set_title(f"Price by {col}")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("boxplot_price_by_category.png", dpi=150)
plt.show()

# 4c. Scatter plot: carat vs price (the strongest expected relationship)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["carat"], y=df["price"], alpha=0.3)
plt.title("Carat vs Price")
plt.xlabel("Carat")
plt.ylabel("Price ($)")
plt.savefig("scatter_carat_vs_price.png", dpi=150)
plt.show()

# 4d. Countplot of categorical variable distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axes, ["cut", "color", "clarity"]):
    sns.countplot(x=df[col], ax=ax, order=df[col].value_counts().index)
    ax.set_title(f"Count of {col}")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("countplot_categoricals.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# Step 5: Observations / insights
# ---------------------------------------------------------
print("""
Observations:
- 'price' and 'carat' are both right-skewed: most diamonds are small/
  cheap, with a long tail of larger, expensive stones.
- The scatter plot of carat vs price shows a clear non-linear
  (roughly exponential) relationship — price increases faster than
  carat weight, consistent with how diamonds are priced in the real
  market (larger stones command a premium beyond a simple linear rate).
- 'x', 'y', and 'z' (the physical dimensions) contain outliers,
  including impossible zero values and a few extreme values (covered
  in the outlier-detection task) that should be cleaned before
  modeling.
- Interestingly, 'Fair' cut diamonds sometimes show a higher median
  price than 'Ideal' cut diamonds in the boxplot — this is because
  cut quality is confounded with carat size in this dataset (larger
  stones are less often cut to 'Ideal' standard), not because worse
  cuts are inherently worth more.
- 'depth' and 'table' are fairly normally distributed and tightly
  clustered, suggesting most diamonds are cut to similar proportions
  regardless of size or price.
""")