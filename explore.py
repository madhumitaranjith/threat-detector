import pandas as pd

# Load the dataset
df = pd.read_parquet("UNSW_NB15_training-set.parquet")

# What type of data is in each column?
print("Column types:")
print(df.dtypes)

# Are there any missing values?
print("\nMissing values per column:")
print(df.isnull().sum())

# Basic stats on the numbers
print("\nBasic statistics:")
print(df.describe())