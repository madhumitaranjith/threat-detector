import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load & prepare (same as before)
df = pd.read_parquet("UNSW_NB15_training-set.parquet")
df = df.drop(columns=['attack_cat'])

cat_columns = ['proto', 'service', 'state']
encoder = LabelEncoder()
for col in cat_columns:
    df[col] = encoder.fit_transform(df[col])

X = df.drop(columns=['label'])
y = df['label']

# Split into training and testing sets
# 80% of data trains the model, 20% tests it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} samples...")
print(f"Testing on {len(X_test)} samples...")

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Test it
y_pred = model.predict(X_test)

# Results
print("\n✅ Model trained!")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed results:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

import joblib

# Save the model
joblib.dump(model, 'model.pkl')
print("\n✅ Model saved as model.pkl!")