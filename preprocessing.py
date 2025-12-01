import pandas as pd
import numpy as np
import os

# === 1. Load dataset safely ===
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "salaries_cyber.csv")

df = pd.read_csv(file_path)
print("File loaded successfully!")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\n--- Sample Data ---")
print(df.head())

# === 2. Basic cleaning (minimal but refined) ===
df.drop_duplicates(inplace=True)

# Drop columns/rows with >50% missing
df.dropna(axis=1, thresh=len(df)*0.5, inplace=True)
df.dropna(axis=0, thresh=len(df.columns)*0.5, inplace=True)

# Identify numeric and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Fill missing values
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# === 3. Data type corrections ===
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === 4. Round numeric columns to remove long decimals ===
# (Change 2 to 0 if you prefer no decimal points at all)
df[num_cols] = df[num_cols].round(2)

# === 5. Clean text columns ===
# Uppercase, strip spaces
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.upper()

# === 6. Handle outliers (clip extreme values) ===
for col in num_cols:
    low, high = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(lower=low, upper=high)

# === 7. Save cleaned dataset ===
clean_path = os.path.join(script_dir, "salaries_cyber_clean.csv")
df.to_csv(clean_path, index=False, float_format="%.2f")  # keep 2 decimal places clean

print("\nCleaning complete!")
print("Saved as:", clean_path)
print("Final shape:", df.shape)
