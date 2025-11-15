import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_data(file_path = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Load the raw dataset from CSV file."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")
    return df


def remove_duplicates(df):
    """Remove duplicate rows from the dataset."""
    print("\nRemoving duplicates...")
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df_cleaned)
    print(f"Removed {duplicates_removed} duplicate row(s)")
    print(f"Data shape after removing duplicates: {df_cleaned.shape}")
    return df_cleaned


def clean_whitespace(df):
    """Remove leading and trailing whitespace from string columns."""
    print("\nCleaning whitespace...")
    df_cleaned = df.copy()
    
    # Strip whitespace from string columns
    string_columns = df_cleaned.select_dtypes(include=['object']).columns
    whitespace_found = False
    
    for col in string_columns:
        # Check if there are any values with leading/trailing whitespace
        if df_cleaned[col].dtype == 'object':
            before = df_cleaned[col].astype(str)
            after = df_cleaned[col].astype(str).str.strip()
            if not before.equals(after):
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                whitespace_found = True
                print(f"  - Cleaned whitespace in {col}")
    
    if not whitespace_found:
        print("  - No whitespace issues found")
    
    return df_cleaned


def fix_data_types(df):
    """Fix data types, especially for columns that should be numeric."""
    print("\nFixing data types...")
    df_cleaned = df.copy()
    
    # Handle TotalCharges - commonly stored as string with empty values
    if 'TotalCharges' in df_cleaned.columns:
        # Replace empty strings with NaN, then convert to numeric
        if df_cleaned['TotalCharges'].dtype == 'object':
            # Count empty strings before conversion
            empty_count = (df_cleaned['TotalCharges'] == '').sum()
            if empty_count > 0:
                print(f"  - Found {empty_count} empty string(s) in TotalCharges")
                # Replace empty strings with NaN
                df_cleaned['TotalCharges'] = df_cleaned['TotalCharges'].replace('', np.nan)
            
            # Convert to numeric, coercing errors to NaN
            df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
            print(f"  - Converted TotalCharges to numeric type")
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    for col in numeric_columns:
        if col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                print(f"  - Converted {col} to numeric type")
    
    return df_cleaned


def validate_data(df):
    """Validate data for logical inconsistencies and invalid values."""
    print("\nValidating data...")
    issues_found = []
    
    # Check for negative values in numeric columns that shouldn't be negative
    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_columns:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues_found.append(f"  ⚠ {col}: {negative_count} negative value(s) found")
    
    # Check for impossible values
    if 'tenure' in df.columns:
        # Tenure should typically be between 0 and 72 (or reasonable max)
        unreasonable_tenure = (df['tenure'] > 100).sum()
        if unreasonable_tenure > 0:
            issues_found.append(f"  ⚠ tenure: {unreasonable_tenure} value(s) > 100")
    
    if 'MonthlyCharges' in df.columns:
        # Monthly charges should be reasonable
        unreasonable_charges = (df['MonthlyCharges'] > 200).sum()
        if unreasonable_charges > 0:
            issues_found.append(f"  ⚠ MonthlyCharges: {unreasonable_charges} value(s) > 200")
    
    # Check for logical inconsistencies
    if 'InternetService' in df.columns and 'OnlineSecurity' in df.columns:
        # If InternetService is "No", then OnlineSecurity should be "No internet service"
        inconsistent = df[(df['InternetService'] == 'No') & 
                           (df['OnlineSecurity'] != 'No internet service')]
        if len(inconsistent) > 0:
            issues_found.append(f"  ⚠ Logical inconsistency: {len(inconsistent)} row(s) with InternetService='No' but OnlineSecurity != 'No internet service'")
    
    if len(issues_found) == 0:
        print("  ✓ No validation issues found")
    else:
        print("  Validation issues found:")
        for issue in issues_found:
            print(issue)
    
    return df


def handle_null_values(df):
    """Replace null values with appropriate defaults based on column type."""
    print("\nHandling null values...")
    
    # Check for null values
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    
    if len(null_columns) == 0:
        print("No null values found in the dataset")
    else:
        print(f"Null values found in {len(null_columns)} column(s):")
        print(null_columns)
    
    # Check for empty strings (which might represent missing data)
    empty_string_counts = (df == '').sum()
    empty_string_columns = empty_string_counts[empty_string_counts > 0]
    
    if len(empty_string_columns) > 0:
        print(f"\nEmpty strings found in {len(empty_string_columns)} column(s):")
        print(empty_string_columns)
    
    df_cleaned = df.copy()
    
    # Special handling for TotalCharges - fill with 0 for new customers (tenure = 0)
    if 'TotalCharges' in df_cleaned.columns and 'tenure' in df_cleaned.columns:
        if df_cleaned['TotalCharges'].isnull().sum() > 0:
            # For new customers (tenure = 0), TotalCharges should be 0
            new_customers_mask = (df_cleaned['tenure'] == 0) & (df_cleaned['TotalCharges'].isnull())
            df_cleaned.loc[new_customers_mask, 'TotalCharges'] = 0
            print(f"  - Filled {new_customers_mask.sum()} TotalCharges null(s) with 0 for new customers")
            
            # For others, fill with median
            remaining_nulls = df_cleaned['TotalCharges'].isnull().sum()
            if remaining_nulls > 0:
                median_value = df_cleaned['TotalCharges'].median()
                df_cleaned['TotalCharges'].fillna(median_value, inplace=True)
                print(f"  - Filled {remaining_nulls} remaining TotalCharges null(s) with median: {median_value}")
    
    # Handle other numeric columns - fill with median
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'TotalCharges' and df_cleaned[col].isnull().sum() > 0:
            median_value = df_cleaned[col].median()
            df_cleaned[col].fillna(median_value, inplace=True)
            print(f"  - Filled {col} (numeric) nulls with median: {median_value}")
    
    # Handle categorical columns - fill with mode (most frequent value)
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_cleaned[col].isnull().sum() > 0:
            mode_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown'
            df_cleaned[col].fillna(mode_value, inplace=True)
            print(f"  - Filled {col} (categorical) nulls with mode: {mode_value}")
        
        # Also handle empty strings in categorical columns
        if (df_cleaned[col] == '').sum() > 0:
            mode_value = df_cleaned[df_cleaned[col] != ''][col].mode()[0] if len(df_cleaned[df_cleaned[col] != ''][col].mode()) > 0 else 'Unknown'
            df_cleaned[col].replace('', mode_value, inplace=True)
            print(f"  - Replaced empty strings in {col} with mode: {mode_value}")
    
    # Verify no nulls remain
    remaining_nulls = df_cleaned.isnull().sum().sum()
    if remaining_nulls == 0:
        print("\n✓ All null values have been handled")
    else:
        print(f"\n⚠ Warning: {remaining_nulls} null values still remain")
    
    return df_cleaned


def generate_data_profile(df):
    """Generate a data quality profile report."""
    print("\nGenerating data profile...")
    
    profile = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'numeric_summary': df.select_dtypes(include=[np.number]).describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {}
    }
    
    # Summary for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        profile['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    print("  ✓ Data profile generated")
    return profile


def save_cleaned_data(df, output_path = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\cleaned_customer_churn.csv"):
    """Save the cleaned dataset to a CSV file."""
    print(f"\nSaving cleaned data to {output_path}...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    df.to_csv(output_path, index=False)
    print(f"✓ Cleaned data saved successfully")
    print(f"Final data shape: {df.shape}")


def prepare_data(input_path, output_path):
    """
    Main function to perform data preparation:
    1. Load data
    2. Clean whitespace
    3. Fix data types
    4. Remove duplicates
    5. Handle null values
    6. Validate data
    7. Generate data profile
    8. Save cleaned data
    
    Parameters:
    -----------
    input_path : str
        Path to the raw data CSV file
    output_path : str
        Path to save the cleaned data CSV file
    
    Returns:
    --------
    df : pandas.DataFrame
        Cleaned dataframe
    profile : dict
        Data quality profile
    """
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    # Load data
    df = load_data(input_path)
    
    # Clean whitespace
    df = clean_whitespace(df)
    
    # Fix data types (especially TotalCharges)
    df = fix_data_types(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Handle null values
    df = handle_null_values(df)
    
    # Validate data
    df = validate_data(df)
    
    # Generate data profile
    profile = generate_data_profile(df)
    
    # Save cleaned data
    save_cleaned_data(df, output_path)
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    
    return df, profile


if __name__ == "__main__":
    # ============================================================
    # CONFIGURATION: Set your file paths here
    # ============================================================
    input_path = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_path = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\cleaned_customer_churn.csv"
    
    # ============================================================
    # RUN DATA PREPARATION PIPELINE
    # ============================================================
    print("\n" + "="*60)
    print("STARTING DATA PREPARATION")
    print("="*60)
    
    # Execute the complete data preparation pipeline
    cleaned_df, profile = prepare_data(input_path, output_path)
    
    # ============================================================
    # DISPLAY FINAL SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"  ✓ Total rows: {len(cleaned_df):,}")
    print(f"  ✓ Total columns: {len(cleaned_df.columns)}")
    print(f"  ✓ Memory usage: {cleaned_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  ✓ Data types: {len(cleaned_df.select_dtypes(include=[np.number]).columns)} numeric, {len(cleaned_df.select_dtypes(include=['object']).columns)} categorical")
    print(f"  ✓ Null values remaining: {cleaned_df.isnull().sum().sum()}")
    print(f"  ✓ Duplicate rows: {cleaned_df.duplicated().sum()}")
    print(f"  ✓ Output saved to: {output_path}")
    print("="*60)
    print("\nData preparation completed successfully! ✓\n")
