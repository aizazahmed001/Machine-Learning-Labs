# Lab Task #01: Find a dataset with outliers, plot it, and identify the outlier
# Dataset: diamonds.csv

import pandas as pd

try:
    import seaborn as sns  # type: ignore[import-not-found]
except ImportError:
    sns = None

import matplotlib.pyplot as plt

if sns is None:
    raise ImportError("seaborn is required for this script. Install it with: pip install seaborn")

# ---------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("diamonds.csv")
print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# Step 2: Summary statistics — spot suspicious values numerically
# ---------------------------------------------------------
print(df.describe())
# Notice: the 'min' of x, y, and z is 0.00, which is physically
# impossible for a real diamond (it has zero width/length/depth).
# Also notice the 'max' of y (58.9) and z (31.8) are far beyond the
# 75th percentile (~6.5 and ~4.0), another sign of outliers.

# ---------------------------------------------------------
# Step 3: Boxplots to visualize outliers in each dimension
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["x", "y", "z"]):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"Boxplot of {col}")
plt.tight_layout()
plt.savefig("boxplots_xyz.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# Step 4: Scatter plot of y vs z to see outliers in context
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["y"], y=df["z"], alpha=0.4)
plt.title("Diamond Width (y) vs Depth (z)")
plt.xlabel("y (mm)")
plt.ylabel("z (mm)")
plt.savefig("scatter_y_vs_z.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# Step 5: Identify the actual outlier rows
# ---------------------------------------------------------
outliers = df[(df["x"] == 0) | (df["y"] == 0) | (df["z"] == 0) |
              (df["y"] > 15) | (df["z"] > 15)]
print(f"\nFound {len(outliers)} outlier rows:")
print(outliers)

# ---------------------------------------------------------
# Step 6: Re-plot the scatter plot with the extreme outliers
# annotated directly on the chart
# ---------------------------------------------------------
extreme_outliers = df[(df["y"] > 15) | (df["z"] > 15)]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["y"], y=df["z"], alpha=0.4, label="Normal points")
sns.scatterplot(x=extreme_outliers["y"], y=extreme_outliers["z"],
                 color="red", s=120, label="Outliers")

for idx, row in extreme_outliers.iterrows():
    plt.annotate(f"idx {idx}\nprice=${row['price']}",
                 (row["y"], row["z"]),
                 textcoords="offset points", xytext=(10, 5),
                 color="red", fontsize=9)

plt.title("Diamond y vs z — Outliers Highlighted")
plt.xlabel("y (mm)")
plt.ylabel("z (mm)")
plt.legend()
plt.tight_layout()
plt.savefig("scatter_outliers_annotated.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# Step 7: Interpretation (print summary for the report)
# ---------------------------------------------------------
print("""
Interpretation:
- Several rows have x, y, or z = 0, which is impossible for a physical
  diamond (a diamond cannot have zero length, width, or depth). These
  are almost certainly data entry errors / missing values encoded as 0.
- A few rows have y or z values far above the normal range
  (e.g. y = 58.9mm or z = 31.8mm, while typical diamonds are 3-9mm).
  Given the carat and price of these rows are unremarkable, these are
  very likely measurement/typing errors (e.g. a misplaced decimal
  point), not genuinely huge diamonds.
- Recommended action before modeling: drop or correct these rows
  rather than leaving them in, since they would distort any model
  trained on these dimension columns.
""")