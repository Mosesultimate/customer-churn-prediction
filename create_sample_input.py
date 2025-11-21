"""Helper script to create a sample input file for predictions."""
import pandas as pd
from pathlib import Path

# Load cleaned data
input_file = r"data\processed\cleaned_customer_churn.csv"
output_file = r"data\predictions\sample_customers.csv"

print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

# Take first 10 customers as sample
sample = df.head(10).copy()

# Remove Churn column if present (not needed for predictions)
if "Churn" in sample.columns:
    sample = sample.drop(columns=["Churn"])
    print("✓ Removed 'Churn' column (not needed for predictions)")

# Create output directory
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

# Save sample
sample.to_csv(output_file, index=False)
print(f"✓ Sample input file created: {output_file}")
print(f"  Contains {len(sample)} customers with {len(sample.columns)} columns")
print(f"\nColumns: {', '.join(sample.columns.tolist())}")
print("\nYou can now run predictions with:")
print(f"  python src/make_predictions.py --input {output_file}")

