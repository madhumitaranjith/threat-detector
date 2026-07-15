import streamlit as st
st.set_page_config(page_title="Threat Detector", page_icon="🛡️", layout="wide")
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('model.pkl')

st.title("AI Network Threat Detector")
st.write("Upload a network traffic file to detect malicious connections.")

# File uploader
uploaded_file = st.file_uploader("Upload your parquet file", type=['parquet'])

if uploaded_file is not None:
    # Load and prepare the data
    df = pd.read_parquet(uploaded_file)

    # Drop attack_cat if it exists
    if 'attack_cat' in df.columns:
        df = df.drop(columns=['attack_cat'])

    # Drop label if it exists
    if 'label' in df.columns:
        labels = df['label']
        df = df.drop(columns=['label'])

    # Encode categories
    cat_columns = ['proto', 'service', 'state']
    encoder = LabelEncoder()
    for col in cat_columns:
        df[col] = encoder.fit_transform(df[col])

    # Make predictions
    predictions = model.predict(df)

    # Show results
    total = len(predictions)
    attacks = predictions.sum()
    normal = total - attacks

    st.subheader("📊 Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Connections", total)
    col2.metric("⚠️ Attacks Detected", attacks)
    col3.metric("✅ Normal Traffic", normal)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.pie(
    [attacks, normal],
    labels=['Attack', 'Normal'],
    colors=['#ff4b4b', '#4bff72'],
    autopct='%1.1f%%'
)
    ax.set_title("Traffic Breakdown")
    st.pyplot(fig)

    # Show the dataframe with predictions
    df['prediction'] = ['🚨 Attack' if p == 1 else '✅ Normal' for p in predictions]
    st.subheader("Connection Details")
    st.dataframe(df)
    