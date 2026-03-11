import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Data Cleaning Agent", layout="wide")

st.title("AI Data Cleaning & Insight Generator")
st.write("Upload a CSV file to profile, clean, and analyze your dataset.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("CSV uploaded successfully.")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

    except Exception as e:
        st.error(f"Error reading file: {e}")