# Detailed Explanation of EDA Notebook Code

This document provides a comprehensive explanation of every code cell in the Exploratory Data Analysis (EDA) notebook, helping you understand what each section does and why it's important.

---

## Table of Contents
1. [Setup and Imports](#1-setup-and-imports)
2. [Loading Data](#2-loading-data)
3. [Basic Data Overview](#3-basic-data-overview)
4. [Target Variable Analysis](#4-target-variable-analysis)
5. [Univariate Analysis](#5-univariate-analysis)
6. [Bivariate Analysis](#6-bivariate-analysis)
7. [Correlation Analysis](#7-correlation-analysis)
8. [Key Insights](#8-key-insights)
9. [Summary Statistics](#9-summary-statistics)

---

## 1. Setup and Imports

### Code:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
```

**Explanation:**
- **pandas (pd)**: Data manipulation and analysis library. Used for reading CSV files, dataframes, and data operations.
- **numpy (np)**: Numerical computing library. Provides mathematical functions and array operations.
- **matplotlib.pyplot (plt)**: Plotting library for creating static visualizations (charts, graphs).
- **seaborn (sns)**: Statistical visualization library built on matplotlib. Provides prettier, more informative plots.
- **warnings**: Used to suppress warning messages that might clutter output.
- **pathlib.Path**: Modern way to handle file paths (cross-platform compatible).

### Display Options:
```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')
```

**Explanation:**
- `display.max_columns = None`: Shows ALL columns when printing dataframes (default is to truncate)
- `display.max_rows = 100`: Shows up to 100 rows when printing (prevents overwhelming output)
- `warnings.filterwarnings('ignore')`: Suppresses warning messages (like deprecation warnings)

### Plotting Style:
```python
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
```

**Explanation:**
- **Try-except blocks**: Handle different matplotlib versions gracefully
- **Style selection**: Sets the visual theme for all plots
  - `seaborn-darkgrid`: Modern style with grid lines
  - `ggplot`: Alternative style inspired by R's ggplot2
- **Why try-except?**: Different matplotlib versions have different style names, so we try multiple options

```python
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
```

**Explanation:**
- `sns.set_palette("husl")`: Sets color palette for seaborn plots (HUSL = perceptually uniform colors)
- `plt.rcParams['figure.figsize']`: Sets default figure size to 12 inches wide × 6 inches tall

---

## 2. Loading Data

### Code:
```python
data_path = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\cleaned_customer_churn.csv"
df = pd.read_csv(data_path)
```

**Explanation:**
- **Raw string (`r"..."`)**: The `r` prefix makes it a "raw string" - backslashes are treated literally (no escape sequences)
  - Without `r`: `"C:\Users"` would try to interpret `\U` as an escape sequence → ERROR
  - With `r`: `r"C:\Users"` treats backslashes as literal characters → Works correctly
- **`pd.read_csv()`**: Reads a CSV file and converts it into a pandas DataFrame
  - DataFrame: A 2D table-like structure (rows and columns) similar to Excel spreadsheet

```python
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head()
```

**Explanation:**
- **`df.shape`**: Returns a tuple `(rows, columns)`
  - `df.shape[0]`: Number of rows
  - `df.shape[1]`: Number of columns
- **`:,` in f-string**: Adds thousand separators (e.g., `7043` becomes `7,043`)
- **`df.head()`**: Displays first 5 rows of the dataframe (default). Useful for quick inspection

---

## 3. Basic Data Overview

### Cell 1: Dataset Information

```python
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
```

**Explanation:**
- **`"=" * 60`**: Creates a string of 60 equal signs (`============================================================`)
- **Purpose**: Visual separator for better readability in output

```python
print(df.dtypes)
```

**Explanation:**
- **`df.dtypes`**: Returns the data type of each column
  - `object`: Usually strings/text
  - `int64`: Integer numbers
  - `float64`: Decimal numbers
  - `bool`: True/False values

```python
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found!")
else:
    print(missing[missing > 0])
```

**Explanation:**
- **`df.isnull()`**: Returns True/False for each cell (True = missing value)
- **`.sum()`**: Counts True values (missing values) per column
- **`missing.sum() == 0`**: Checks if total missing values is zero
- **`missing[missing > 0]`**: Filters to show only columns with missing values

```python
print(f"Duplicate Rows: {df.duplicated().sum()}")
```

**Explanation:**
- **`df.duplicated()`**: Returns True for rows that are exact duplicates of previous rows
- **`.sum()`**: Counts how many duplicate rows exist

```python
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

**Explanation:**
- **`df.memory_usage(deep=True)`**: Calculates memory used by each column
  - `deep=True`: Includes memory for object types (strings)
- **`.sum()`**: Total memory for entire dataframe
- **`/ 1024**2`**: Converts bytes to megabytes (1024² = 1,048,576 bytes = 1 MB)
- **`:.2f`**: Formats to 2 decimal places

### Cell 2: Numeric Columns Summary

```python
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'customerID' in numeric_cols:
    numeric_cols.remove('customerID')
print(df[numeric_cols].describe())
```

**Explanation:**
- **`df.select_dtypes(include=[np.number])`**: Filters dataframe to only numeric columns
- **`.columns.tolist()`**: Converts column names to a Python list
- **`if 'customerID' in numeric_cols`**: Checks if customerID was incorrectly identified as numeric
- **`numeric_cols.remove('customerID')`**: Removes it from analysis (ID columns aren't useful for statistics)
- **`df[numeric_cols].describe()`**: Generates descriptive statistics:
  - **count**: Number of non-null values
  - **mean**: Average value
  - **std**: Standard deviation (measure of spread)
  - **min**: Minimum value
  - **25%**: First quartile (25% of values are below this)
  - **50%**: Median (middle value)
  - **75%**: Third quartile (75% of values are below this)
  - **max**: Maximum value

### Cell 3: Categorical Columns Summary

```python
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'customerID' in categorical_cols:
    categorical_cols.remove('customerID')

for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Value counts:")
    print(df[col].value_counts().head(10))
```

**Explanation:**
- **`df.select_dtypes(include=['object'])`**: Selects string/categorical columns
- **`df[col].nunique()`**: Counts number of unique values in the column
- **`df[col].value_counts()`**: Counts frequency of each unique value
- **`.head(10)`**: Shows top 10 most frequent values
- **Purpose**: Understand what categories exist and their distribution

---

## 4. Target Variable Analysis (Churn)

### Code:
```python
if 'Churn' in df.columns:
    churn_counts = df['Churn'].value_counts()
    churn_percent = df['Churn'].value_counts(normalize=True) * 100
```

**Explanation:**
- **`if 'Churn' in df.columns`**: Safety check - only runs if Churn column exists
- **`df['Churn'].value_counts()`**: Counts occurrences of each value (Yes/No)
- **`normalize=True`**: Converts counts to proportions (0.0 to 1.0)
- **`* 100`**: Converts proportions to percentages (0.0-1.0 → 0-100%)

### Visualization Part 1: Bar Chart

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
```

**Explanation:**
- **`plt.subplots(1, 2, ...)`**: Creates a figure with 1 row, 2 columns (side-by-side plots)
- **`figsize=(14, 5)`**: Figure size: 14 inches wide × 5 inches tall
- **`fig`**: The entire figure container
- **`axes`**: Array of subplot objects (axes[0] and axes[1])

```python
churn_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
```

**Explanation:**
- **`.plot(kind='bar', ...)`**: Creates a bar chart
- **`ax=axes[0]`**: Plots on the first subplot (left side)
- **`color=['#2ecc71', '#e74c3c']`**: Custom colors (green for "No", red for "Yes")
  - Hex colors: `#2ecc71` = green, `#e74c3c` = red

```python
axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Churn', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
```

**Explanation:**
- **`.set_title()`**: Sets the plot title
- **`fontsize`**: Text size
- **`fontweight='bold'`**: Makes text bold
- **`.set_xlabel()` / `.set_ylabel()`**: Labels for axes

```python
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 100, str(v), ha='center', fontweight='bold')
```

**Explanation:**
- **`enumerate()`**: Gets both index (i) and value (v) from the counts
- **`axes[0].text(i, v + 100, str(v), ...)`**: Adds text label on each bar
  - `i`: x-position (bar index)
  - `v + 100`: y-position (slightly above the bar)
  - `str(v)`: The count value as text
  - `ha='center'`: Horizontal alignment centered

### Visualization Part 2: Pie Chart

```python
axes[1].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'], startangle=90)
```

**Explanation:**
- **`.pie()`**: Creates a pie chart
- **`churn_counts.values`**: The data values (counts)
- **`labels=churn_counts.index`**: Labels for each slice (Yes/No)
- **`autopct='%1.1f%%'`**: Shows percentage on each slice (1 decimal place)
- **`startangle=90`**: Rotates pie so it starts at the top

```python
plt.tight_layout()
plt.show()
```

**Explanation:**
- **`plt.tight_layout()`**: Automatically adjusts spacing between subplots
- **`plt.show()`**: Displays the figure

### Class Imbalance Check

```python
print(f"\n⚠ Class Imbalance: {churn_percent.iloc[0]:.1f}% vs {churn_percent.iloc[1]:.1f}%")
```

**Explanation:**
- **`.iloc[0]`**: Gets first value (index 0)
- **`.iloc[1]`**: Gets second value (index 1)
- **Purpose**: Identifies if classes are imbalanced (e.g., 80% vs 20%)
  - **Imbalanced**: One class much larger than the other (problem for machine learning)
  - **Balanced**: Classes are roughly equal (50/50 or 60/40)

---

## 5. Univariate Analysis - Numeric Features

### Code Structure:
```python
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
numeric_features = [col for col in numeric_features if col in df.columns]
```

**Explanation:**
- **List of expected features**: Defines which numeric columns to analyze
- **List comprehension**: Filters to only include columns that actually exist in the dataframe
  - `[col for col in numeric_features if col in df.columns]`
  - Reads as: "For each col in numeric_features, include it if it exists in df.columns"

### Creating Subplots:
```python
fig, axes = plt.subplots(len(numeric_features), 2, figsize=(14, 5*len(numeric_features)))
```

**Explanation:**
- **`len(numeric_features)` rows**: One row per feature
- **`2 columns**: Histogram and box plot side-by-side
- **`figsize=(14, 5*len(numeric_features))`**: Height scales with number of features

```python
if len(numeric_features) == 1:
    axes = axes.reshape(1, -1)
```

**Explanation:**
- **`.reshape(1, -1)`**: Converts 1D array to 2D array (1 row, auto columns)
- **Why needed?**: When there's only 1 feature, `axes` is 1D, but we need 2D for indexing `axes[idx, 0]`

### Histogram Creation:
```python
axes[idx, 0].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
```

**Explanation:**
- **`.hist()`**: Creates a histogram (distribution plot)
- **`df[col].dropna()`**: Removes missing values before plotting
- **`bins=30`**: Divides data into 30 intervals (bars)
- **`edgecolor='black'`**: Black outline on each bar
- **`alpha=0.7`**: Transparency (0.7 = 70% opaque, 30% transparent)

```python
axes[idx, 0].axvline(df[col].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df[col].mean():.2f}')
```

**Explanation:**
- **`.axvline()`**: Draws a vertical line
- **`df[col].mean()`**: X-position (mean value)
- **`color='red'`**: Red color
- **`linestyle='--'`**: Dashed line
- **Purpose**: Visual reference to show where the average is

### Box Plot:
```python
axes[idx, 1].boxplot(df[col].dropna(), vert=True)
```

**Explanation:**
- **`.boxplot()`**: Creates a box plot (shows distribution, quartiles, outliers)
- **`vert=True`**: Vertical orientation
- **Box plot shows**:
  - **Box**: Interquartile range (25th to 75th percentile)
  - **Line in box**: Median (50th percentile)
  - **Whiskers**: Range of data (excluding outliers)
  - **Dots**: Outliers (unusual values)

---

## 6. Bivariate Analysis - Features vs Churn

### Numeric Features vs Churn:

```python
df.boxplot(column=col, by='Churn', ax=axes[idx], grid=False)
```

**Explanation:**
- **`column=col`**: Which numeric column to plot
- **`by='Churn'`**: Creates separate box plots for each Churn category (Yes/No)
- **`ax=axes[idx]`**: Which subplot to use
- **Purpose**: Compare distributions between churned and non-churned customers

### Statistical Comparison:
```python
churned = df[df['Churn'] == 'Yes'][col]
not_churned = df[df['Churn'] == 'No'][col]
```

**Explanation:**
- **`df[df['Churn'] == 'Yes']`**: Filters dataframe to only rows where Churn is "Yes"
- **`[col]`**: Extracts just that column
- **Result**: Two series (arrays) - one for churned, one for not churned

```python
print(f"  Churned:     Mean={churned.mean():.2f}, Median={churned.median():.2f}")
print(f"  Not Churned: Mean={not_churned.mean():.2f}, Median={not_churned.median():.2f}")
print(f"  Difference:  {abs(churned.mean() - not_churned.mean()):.2f}")
```

**Explanation:**
- **`.mean()`**: Calculates average
- **`.median()`**: Calculates middle value
- **`abs(...)`**: Absolute value (always positive)
- **Purpose**: Quantify the difference between groups

### Categorical Features vs Churn:

```python
crosstab = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
```

**Explanation:**
- **`pd.crosstab()`**: Creates a cross-tabulation (contingency table)
- **`df[col], df['Churn']`**: Rows and columns of the table
- **`normalize='index'`**: Converts counts to percentages (normalized by row)
- **`* 100`**: Converts to percentage (0.0-1.0 → 0-100%)
- **Result**: Table showing churn rate for each category

**Example Output:**
```
Contract        No    Yes
Month-to-month  45%   55%
One year        90%   10%
Two year        97%    3%
```

### Visualization:
```python
crosstab_pct.plot(kind='bar', ax=axes[row, col_idx], 
                 color=['#2ecc71', '#e74c3c'], width=0.8)
```

**Explanation:**
- **`kind='bar'`**: Stacked bar chart
- **`width=0.8`**: Bar width (80% of available space)
- **Purpose**: Visual comparison of churn rates across categories

---

## 7. Correlation Analysis

### Correlation Matrix:
```python
corr_matrix = df[numeric_cols].corr()
```

**Explanation:**
- **`.corr()`**: Calculates correlation coefficient between all pairs of numeric columns
- **Correlation coefficient**: Measures linear relationship strength
  - **+1.0**: Perfect positive correlation (as one increases, other increases)
  - **0.0**: No correlation
  - **-1.0**: Perfect negative correlation (as one increases, other decreases)
  - **> 0.7 or < -0.7**: Strong correlation
  - **0.3 to 0.7 or -0.3 to -0.7**: Moderate correlation
  - **< 0.3 and > -0.3**: Weak correlation

### Heatmap:
```python
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
```

**Explanation:**
- **`sns.heatmap()`**: Creates a color-coded matrix
- **`annot=True`**: Shows correlation values in each cell
- **`fmt='.2f'`**: Formats numbers to 2 decimal places
- **`cmap='coolwarm'`**: Color scheme (blue = negative, red = positive)
- **`center=0`**: Centers color scale at 0
- **`square=True`**: Makes cells square-shaped
- **`linewidths=1`**: Border width between cells
- **`cbar_kws={"shrink": 0.8}`**: Makes color bar 80% of default size

### Correlation with Churn:
```python
df_encoded = df.copy()
df_encoded['Churn_encoded'] = (df_encoded['Churn'] == 'Yes').astype(int)
```

**Explanation:**
- **`.copy()`**: Creates a copy (doesn't modify original dataframe)
- **`(df_encoded['Churn'] == 'Yes')`**: Creates boolean series (True/False)
- **`.astype(int)`**: Converts True→1, False→0
- **Why needed?**: Correlation requires numeric values, so we convert "Yes"/"No" to 1/0

```python
churn_corr = df_encoded[numeric_cols + ['Churn_encoded']].corr()['Churn_encoded'].sort_values(ascending=False)
```

**Explanation:**
- **`numeric_cols + ['Churn_encoded']`**: Combines lists (all numeric columns + encoded churn)
- **`.corr()['Churn_encoded']`**: Gets correlation of all features with Churn_encoded
- **`.sort_values(ascending=False)`**: Sorts from highest to lowest correlation
- **Purpose**: Identifies which features are most predictive of churn

---

## 8. Key Insights

### Overall Churn Rate:
```python
overall_churn = (df['Churn'] == 'Yes').mean() * 100
```

**Explanation:**
- **`(df['Churn'] == 'Yes')`**: Boolean series (True where Churn is "Yes")
- **`.mean()`**: Calculates proportion (True = 1, False = 0, so mean = proportion of True)
- **`* 100`**: Converts to percentage

### Top Risk Factors:
```python
for col in categorical_features:
    churn_by_category = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    max_churn = churn_by_category.max()
    risk_factors[col] = (churn_by_category.idxmax(), max_churn)
```

**Explanation:**
- **`df.groupby(col)['Churn']`**: Groups data by category, then looks at Churn column
- **`.apply(lambda x: (x == 'Yes').mean() * 100)`**: For each group, calculates churn percentage
  - `lambda x`: Anonymous function (x is the Churn values for that group)
  - `(x == 'Yes').mean()`: Proportion of "Yes" in that group
- **`.max()`**: Highest churn rate
- **`.idxmax()`**: Which category has the highest churn rate
- **Purpose**: Identifies which category values are most risky

### Tenure Insights:
```python
low_tenure_churn = df[df['tenure'] <= 12]['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
```

**Explanation:**
- **`df[df['tenure'] <= 12]`**: Filters to customers with tenure ≤ 12 months
- **`['Churn']`**: Gets Churn column
- **`.apply(lambda x: (x == 'Yes').mean() * 100)`**: Calculates churn percentage
- **Purpose**: Compare churn rates between new customers (≤12 months) and established customers (>12 months)

---

## 9. Summary Statistics

### Comparative Statistics:
```python
comparison = df.groupby('Churn')[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
```

**Explanation:**
- **`df.groupby('Churn')`**: Groups data by Churn status (Yes/No)
- **`[numeric_cols]`**: Selects only numeric columns
- **`.agg(['mean', 'median', 'std', 'min', 'max'])`**: Applies multiple aggregation functions
  - **mean**: Average
  - **median**: Middle value
  - **std**: Standard deviation (spread)
  - **min**: Minimum value
  - **max**: Maximum value
- **Result**: Table comparing statistics between churned and non-churned groups

### Overlay Histograms:
```python
axes[idx].hist(not_churned, bins=30, alpha=0.6, label='No Churn', color='#2ecc71', edgecolor='black')
axes[idx].hist(churned, bins=30, alpha=0.6, label='Churn', color='#e74c3c', edgecolor='black')
```

**Explanation:**
- **Two histograms on same plot**: Overlays distributions
- **`alpha=0.6`**: 60% opacity (allows seeing both distributions)
- **`label='...'`**: Text for legend
- **Purpose**: Visual comparison of distributions between groups

---

## Key Concepts Summary

### Data Types:
- **Numeric (int/float)**: Numbers that can be used in calculations
- **Categorical (object)**: Text/categories (Yes/No, Contract types, etc.)

### Statistical Measures:
- **Mean**: Average value
- **Median**: Middle value (50th percentile)
- **Standard Deviation**: Measure of spread/variability
- **Quartiles**: 25th, 50th (median), 75th percentiles

### Visualization Types:
- **Bar Chart**: Compare counts/categories
- **Histogram**: Show distribution of numeric data
- **Box Plot**: Show distribution, quartiles, outliers
- **Pie Chart**: Show proportions/percentages
- **Heatmap**: Show correlation matrix
- **Overlay Histograms**: Compare two distributions

### Important Functions:
- **`.value_counts()`**: Count frequency of values
- **`.groupby()`**: Group data by category
- **`.corr()`**: Calculate correlations
- **`pd.crosstab()`**: Create contingency tables
- **`.describe()`**: Generate summary statistics

---

## How to Explain This Code

### When Presenting:
1. **Start with the goal**: "We're exploring the data to understand patterns and relationships"
2. **Explain each section's purpose**: "This section analyzes the target variable..."
3. **Show the output**: "As you can see, the churn rate is..."
4. **Interpret results**: "This means customers with month-to-month contracts have higher churn"
5. **Connect to next steps**: "Based on these insights, we'll create features for modeling"

### Key Points to Emphasize:
- **EDA is exploratory**: We're discovering patterns, not confirming hypotheses
- **Visualizations make patterns obvious**: Charts reveal what numbers hide
- **Each analysis answers a question**: "Which customers churn more?" "What features matter?"
- **Insights guide modeling**: Understanding data helps build better models

---

## Common Questions & Answers

**Q: Why do we check for missing values if data is already cleaned?**
A: Verification step - confirms the cleaning process worked correctly.

**Q: Why create so many visualizations?**
A: Different visualizations reveal different insights. Box plots show outliers, histograms show distributions, bar charts show comparisons.

**Q: What does correlation tell us?**
A: Correlation shows which numeric features move together. High correlation with churn means that feature is predictive.

**Q: Why analyze categorical features separately?**
A: Categorical features need different analysis (crosstabs, not correlations). They show which categories have higher churn rates.

**Q: What's the purpose of the insights section?**
A: Summarizes key findings in plain language, making it easy to understand what matters most for predicting churn.

---

This explanation should help you understand and explain every part of the EDA notebook code!

