# Lab Task: Advanced Visualizations on a Subset of Variables
# Dataset: diamonds.csv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("diamonds.csv")

# ---------------------------------------------------------
# Step 2: Select a subset of variables
# ---------------------------------------------------------
# We choose carat, price, and the three physical dimensions
# (x, y, z), since these are the core drivers of a diamond's
# value and are all numeric/continuous, making them ideal for
# correlation and multivariate analysis.
subset = df[["carat", "price", "x", "y", "z"]]

print("Selected subset:")
print(subset.head())
print("\nSummary statistics:")
print(subset.describe())

# ---------------------------------------------------------
# Step 3a: Heatmap - correlation between chosen variables
# ---------------------------------------------------------
plt.figure(figsize=(7, 6))
corr = subset.corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation Heatmap: carat, price, x, y, z")
plt.tight_layout()
plt.savefig("heatmap_subset.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# Step 3b: Pairwise scatterplots
# ---------------------------------------------------------
# Note: we filter out the known outlier rows (0 or extreme x/y/z
# values, identified in the earlier outlier-detection task) so the
# pairplot isn't dominated/distorted by a handful of bad data points.
clean_subset = subset[
    (subset["x"] > 0) & (subset["y"] > 0) & (subset["z"] > 0) &
    (subset["y"] < 15) & (subset["z"] < 15)
]

pairplot = sns.pairplot(clean_subset, diag_kind="hist", plot_kws={"alpha": 0.3, "s": 15})
pairplot.fig.suptitle("Pairwise Relationships: carat, price, x, y, z", y=1.02)
pairplot.savefig("pairplot_subset.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# Step 3c: Time series plot
# ---------------------------------------------------------
# Not applicable - this dataset has no date/time column (each row
# is a single diamond listing with no timestamp), so a time series
# plot cannot be meaningfully created here. This is noted rather
# than fabricated.
print("\nNote: No date/time variable exists in this dataset, so a "
      "time series plot is not applicable here.")

# ---------------------------------------------------------
# Step 4: Interpretation
# ---------------------------------------------------------
print("""
Interpretation:

Heatmap:
- carat, x, y, and z are extremely highly correlated with each other
  (all above 0.95). This makes sense: x, y, z are the physical
  dimensions of the stone, and carat (weight) is a direct function
  of its volume/size. This is a case of severe multicollinearity -
  if building a predictive model, using all four together as
  features would be redundant.
- price correlates strongly with carat (~0.92) and similarly with
  x, y, z - confirming that size is the single biggest driver of a
  diamond's price, more so than cut, color, or clarity individually.

Pairwise scatterplots:
- carat vs price shows the same curved, non-linear growth pattern
  seen earlier - price accelerates faster than carat weight increases.
- x vs y and x vs z form very tight, nearly straight diagonal lines,
  visually confirming the near-perfect correlation from the heatmap -
  diamonds are cut to fairly consistent proportions regardless of size.
- The diagonal histograms show carat and price are right-skewed
  (many small/cheap diamonds, fewer large/expensive ones), while x,
  y, z are closer to a normal-ish shape once outliers are removed.
- No obvious multi-cluster grouping appears - the data forms one
  continuous trend rather than distinct sub-populations, suggesting
  diamond pricing follows a fairly consistent underlying rule based
  on size.

Overall takeaway:
- Physical size (captured almost interchangeably by carat, x, y, z)
  is the dominant factor in diamond pricing. Any model built on this
  data should be careful about including all of carat/x/y/z together
  due to multicollinearity, and may benefit from a non-linear model
  or a log-transform of price/carat to handle the curved relationship.
""")