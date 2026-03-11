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

        st.subheader("Column Names")
        st.write(list(df.columns))

        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values
        })
        st.dataframe(dtype_df)

        st.subheader("Missing Values Per Column")
        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values
        })
        st.dataframe(missing_df)

        st.subheader("Duplicate Rows")
        duplicate_count = df.duplicated().sum()
        st.write(f"Duplicate rows: {duplicate_count}")

        st.subheader("Cleaning Options")

        if st.button("Clean Dataset"):
            cleaned_df = df.copy()

            original_rows = cleaned_df.shape[0]
            original_missing = cleaned_df.isnull().sum().sum()

            # Fill missing values
            for col in cleaned_df.columns:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                else:
                    if cleaned_df[col].mode().empty:
                        cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

            # Remove duplicates
            cleaned_df = cleaned_df.drop_duplicates()

            cleaned_rows = cleaned_df.shape[0]
            cleaned_missing = cleaned_df.isnull().sum().sum()
            removed_duplicates = original_rows - cleaned_rows

            st.success("Dataset cleaned successfully.")

            st.subheader("Cleaning Summary")
            st.write(f"Original total missing values: {original_missing}")
            st.write(f"Remaining missing values: {cleaned_missing}")
            st.write(f"Duplicate rows removed: {removed_duplicates}")
            st.write(f"Final row count: {cleaned_rows}")

            st.subheader("Cleaned Dataset Preview")
            st.dataframe(cleaned_df.head(20))  # show first 20 rows as preview

            st.subheader("Full Cleaned Dataset")
            st.dataframe(cleaned_df)  # shows entire cleaned dataframe in Streamlit table

            # Convert full cleaned dataset to CSV for download
            csv = cleaned_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Full Cleaned Dataset",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error reading file: {e}")