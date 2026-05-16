# ==========================================
# COMPLETE DATA CLEANING CODE
# ==========================================

# STEP 1 — Import Required Libraries
import pandas as pd
import numpy as np

print("Libraries Imported Successfully")


# STEP 2 — Load Dataset
df = pd.read_csv("airquality.csv")

print("\nDataset Loaded Successfully")


# STEP 3 — Display First 5 Rows
print("\nFirst 5 Rows of Dataset:")
print(df.head())


# STEP 4 — Check Dataset Shape
print("\nDataset Shape:")
print(df.shape)


# STEP 5 — Display Dataset Information
print("\nDataset Information:")
print(df.info())


# STEP 6 — Check Missing Values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


# STEP 7 — Check Duplicate Rows
duplicates = df.duplicated().sum()

print("\nNumber of Duplicate Rows:", duplicates)


# STEP 8 — Remove Duplicate Rows
df = df.drop_duplicates()

print("\nDuplicate Rows Removed Successfully")


# STEP 9 — Standardize Column Names
df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

print("\nStandardized Column Names:")
print(df.columns)


# STEP 10 — Separate Numeric and Categorical Columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns


# STEP 11 — Fill Missing Values in Numeric Columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nNumeric Missing Values Filled Successfully")


# STEP 12 — Fill Missing Values in Categorical Columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nCategorical Missing Values Filled Successfully")


# STEP 13 — Remove Extra Spaces from Text Columns
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

print("\nExtra Spaces Removed Successfully")


# STEP 14 — Check Missing Values After Cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# STEP 15 — Final Dataset Information
print("\nFinal Dataset Information:")
print(df.info())


# STEP 16 — Display Cleaned Dataset
print("\nCleaned Dataset Preview:")
print(df.head())


# STEP 17 — Save Cleaned Dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned Dataset Saved Successfully as 'cleaned_dataset.csv'")