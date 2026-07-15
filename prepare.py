import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load
df = pd.read_parquet("UNSW_NB15_training-set.parquet")

# Drop attack_cat - we don't need it
df = df.drop(columns=['attack_cat'])

# Convert category columns to numbers
cat_columns = ['proto', 'service', 'state']
encoder = LabelEncoder()
for col in cat_columns:
    df[col] = encoder.fit_transform(df[col])

# Split into features (X) and label (y)
X = df.drop(columns=['label'])
y = df['label']

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("\nFirst 5 rows of features:")
print(X.head())
print("\nLabel distribution:")
print(y.value_counts())
