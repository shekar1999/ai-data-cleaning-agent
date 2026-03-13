import streamlit as st
import pandas as pd
import json
from io import StringIO
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Data Parsing, Cleaning & Insight Generator", layout="wide")

st.title("AI Data Parsing, Cleaning & Insight Generator")
st.write(
    "Upload CSV, Excel, TSV, TXT, JSON, or DATA files to parse, clean, and analyze your dataset."
)

# -------------------------------
# Sidebar Options
# -------------------------------
st.sidebar.header("Upload Settings")

has_header = st.sidebar.checkbox("File contains header row", value=True)

manual_delimiter = st.sidebar.selectbox(
    "Delimiter for TXT / DATA / TSV files",
    options=["Auto Detect", "Comma (,)", "Tab (\\t)", "Pipe (|)", "Semicolon (;)", "Space"],
    index=0
)

remove_duplicates = st.sidebar.checkbox("Remove exact duplicate rows during cleaning", value=False)

uploaded_file = st.file_uploader(
    "Upload your data file",
    type=["csv", "xlsx", "txt", "tsv", "json", "data"]
)

# -------------------------------
# Helper Functions
# -------------------------------
def get_separator(delimiter_choice):
    if delimiter_choice == "Comma (,)":
        return ","
    elif delimiter_choice == "Tab (\\t)":
        return "\t"
    elif delimiter_choice == "Pipe (|)":
        return "|"
    elif delimiter_choice == "Semicolon (;)":
        return ";"
    elif delimiter_choice == "Space":
        return r"\s+"
    return None


def load_file(file, has_header=True, delimiter_choice="Auto Detect"):
    file_name = file.name.lower()
    header_option = 0 if has_header else None
    sep = get_separator(delimiter_choice)

    if file_name.endswith(".csv"):
        if sep:
            if sep == r"\s+":
                return pd.read_csv(file, sep=sep, header=header_option, engine="python", on_bad_lines="skip")
            return pd.read_csv(file, sep=sep, header=header_option, on_bad_lines="skip")
        return pd.read_csv(file, header=header_option, on_bad_lines="skip")

    elif file_name.endswith(".xlsx"):
        return pd.read_excel(file, header=header_option)

    elif file_name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t", header=header_option, on_bad_lines="skip")

    elif file_name.endswith(".txt") or file_name.endswith(".data"):
        text_data = file.read().decode("utf-8", errors="ignore")

        if sep:
            return pd.read_csv(
                StringIO(text_data),
                sep=sep,
                header=header_option,
                engine="python",
                on_bad_lines="skip"
            )

        delimiters_to_try = [",", "\t", "|", ";", r"\s+"]
        for trial_sep in delimiters_to_try:
            try:
                df = pd.read_csv(
                    StringIO(text_data),
                    sep=trial_sep,
                    header=header_option,
                    engine="python",
                    on_bad_lines="skip"
                )
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue

        raise ValueError(
            "Could not reliably parse TXT/DATA file. Please choose the delimiter manually from the sidebar."
        )

    elif file_name.endswith(".json"):
        raw_data = json.load(file)

        if isinstance(raw_data, list):
            return pd.DataFrame(raw_data)
        elif isinstance(raw_data, dict):
            return pd.json_normalize(raw_data)
        else:
            raise ValueError("Unsupported JSON structure.")

    else:
        raise ValueError("Unsupported file format.")


def build_recommendations(df):
    recommendations = []
    duplicate_count = df.duplicated().sum()

    for col in df.columns:
        missing_count = df[col].isnull().sum()

        if missing_count > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                recommendations.append({
                    "Column": col,
                    "Issue": f"{missing_count} missing values",
                    "Recommendation": "Fill with median"
                })
            else:
                recommendations.append({
                    "Column": col,
                    "Issue": f"{missing_count} missing values",
                    "Recommendation": "Fill with mode"
                })

    if duplicate_count > 0:
        recommendations.append({
            "Column": "All Rows",
            "Issue": f"{duplicate_count} exact duplicate rows found",
            "Recommendation": "Review before removal; repeated numeric rows may be valid observations"
        })

    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            original_non_null = df[col].notnull().sum()
            converted_non_null = converted.notnull().sum()

            if original_non_null > 0 and converted_non_null / original_non_null > 0.8:
                recommendations.append({
                    "Column": col,
                    "Issue": "Object column appears mostly numeric",
                    "Recommendation": "Convert to numeric datatype"
                })

    return recommendations


def calculate_quality_score(df):
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    total_duplicates = df.duplicated().sum()

    missing_penalty = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    duplicate_penalty = (total_duplicates / df.shape[0]) * 100 if df.shape[0] > 0 else 0

    raw_score = 100 - (missing_penalty * 0.7 + duplicate_penalty * 0.3)
    return max(0, round(raw_score, 2))


def clean_dataset(df, remove_duplicates=False):
    cleaned_df = df.copy()

    before_rows = cleaned_df.shape[0]
    before_cols = cleaned_df.shape[1]
    before_missing = cleaned_df.isnull().sum().sum()
    before_duplicates = cleaned_df.duplicated().sum()

    datatype_fixes = []

    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == "object":
            converted = pd.to_numeric(cleaned_df[col], errors="coerce")
            original_non_null = cleaned_df[col].notnull().sum()
            converted_non_null = converted.notnull().sum()

            if original_non_null > 0 and converted_non_null / original_non_null > 0.8:
                cleaned_df[col] = converted
                datatype_fixes.append(col)

    for col in cleaned_df.columns:
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            median_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_value)
        else:
            if cleaned_df[col].mode().empty:
                cleaned_df[col] = cleaned_df[col].fillna("Unknown")
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

    removed_duplicates = 0
    if remove_duplicates:
        rows_before_drop = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = rows_before_drop - cleaned_df.shape[0]

    after_rows = cleaned_df.shape[0]
    after_cols = cleaned_df.shape[1]
    after_missing = cleaned_df.isnull().sum().sum()
    after_duplicates = cleaned_df.duplicated().sum()

    results = {
        "cleaned_df": cleaned_df,
        "datatype_fixes": datatype_fixes,
        "before_rows": before_rows,
        "before_cols": before_cols,
        "before_missing": before_missing,
        "before_duplicates": before_duplicates,
        "after_rows": after_rows,
        "after_cols": after_cols,
        "after_missing": after_missing,
        "after_duplicates": after_duplicates,
        "removed_duplicates": removed_duplicates
    }

    return results


def detect_anomalies(df):
    numeric_df = df.select_dtypes(include=["number"]).copy()

    if numeric_df.shape[1] == 0 or numeric_df.shape[0] < 5:
        return None, None

    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.shape[1] == 0:
        return None, None
        
    numeric_df = numeric_df.fillna(numeric_df.median())

    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(numeric_df)

    result_df = df.copy()
    result_df["anomaly_flag"] = preds
    anomalies = result_df[result_df["anomaly_flag"] == -1]

    return result_df, anomalies


# -------------------------------
# Main App
# -------------------------------
if uploaded_file is not None:
    try:
        df = load_file(uploaded_file, has_header=has_header, delimiter_choice=manual_delimiter)

        if not has_header:
            df.columns = [f"column_{i+1}" for i in range(df.shape[1])]

        if df.empty:
            st.warning("The uploaded file was parsed, but it contains no usable data.")
        else:
            st.success(f"{uploaded_file.name} uploaded and parsed successfully.")

            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)

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
            st.dataframe(dtype_df, use_container_width=True)

            st.subheader("Missing Values Per Column")
            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing Values": df.isnull().sum().values
            })
            st.dataframe(missing_df, use_container_width=True)

            st.subheader("Duplicate Rows")
            duplicate_count = df.duplicated().sum()
            st.write(f"Exact duplicate rows detected: {duplicate_count}")

            if duplicate_count > 0:
                st.info("Duplicates are not removed automatically. Repeated numeric rows may be valid observations.")

            st.subheader("Cleaning Recommendations")
            recommendations = build_recommendations(df)

            if recommendations:
                recommendations_df = pd.DataFrame(recommendations)
                st.dataframe(recommendations_df, use_container_width=True)
            else:
                st.success("No major cleaning recommendations. Dataset looks good.")

            st.subheader("Data Quality Score")
            quality_score = calculate_quality_score(df)
            st.metric(label="Overall Data Quality Score", value=f"{quality_score}/100")

            if quality_score >= 90:
                st.success("Excellent data quality.")
            elif quality_score >= 75:
                st.info("Good data quality, but some cleaning is recommended.")
            elif quality_score >= 50:
                st.warning("Moderate data quality issues detected.")
            else:
                st.error("Poor data quality. Significant cleaning needed.")

            # Charts
            st.subheader("Visual Analysis")
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if numeric_cols:
                selected_col = st.selectbox("Select a numeric column for histogram", numeric_cols)
                fig, ax = plt.subplots(figsize=(8, 4))
                valid_data = df[selected_col].dropna()
                if valid_data.empty:
                    st.warning("No valid data to plot in this column.")
                else:
                    ax.hist(valid_data, bins=20)
                    ax.set_title(f"Distribution of {selected_col}")
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)

                if len(numeric_cols) >= 2:
                    st.subheader("Correlation Heatmap")
                    corr = df[numeric_cols].corr()
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax2)
                    st.pyplot(fig2)
            else:
                st.info("No numeric columns available for charting.")

            # Anomaly detection
            st.subheader("Anomaly Detection")
            if st.button("Detect Anomalies"):
                anomaly_df, anomalies = detect_anomalies(df)

                if anomaly_df is None:
                    st.warning("Not enough numeric data available for anomaly detection.")
                else:
                    st.success("Anomaly detection completed.")
                    st.write(f"Anomalies detected: {len(anomalies)}")
                    st.dataframe(anomalies.head(20), use_container_width=True)

            # Cleaning
            st.subheader("Cleaning Options")
            st.write(f"Remove exact duplicates during cleaning: {'Yes' if remove_duplicates else 'No'}")

            if st.button("Clean Dataset"):
                results = clean_dataset(df, remove_duplicates=remove_duplicates)
                cleaned_df = results["cleaned_df"]

                st.success("Dataset cleaned successfully.")

                st.subheader("Before vs After Comparison")
                comparison_df = pd.DataFrame({
                    "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
                    "Before Cleaning": [
                        results["before_rows"],
                        results["before_cols"],
                        results["before_missing"],
                        results["before_duplicates"]
                    ],
                    "After Cleaning": [
                        results["after_rows"],
                        results["after_cols"],
                        results["after_missing"],
                        results["after_duplicates"]
                    ]
                })
                st.dataframe(comparison_df, use_container_width=True)

                st.subheader("Cleaning Summary")
                st.write(f"Duplicate rows removed: {results['removed_duplicates']}")

                if results["datatype_fixes"]:
                    st.write("Datatype-fixed columns:")
                    st.write(results["datatype_fixes"])
                else:
                    st.write("No columns required datatype conversion.")

                st.subheader("Cleaned Dataset Preview")
                st.dataframe(cleaned_df.head(20), use_container_width=True)

                st.subheader("Full Cleaned Dataset")
                st.dataframe(cleaned_df, use_container_width=True)

                csv = cleaned_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Full Cleaned Dataset",
                    data=csv,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error("Parsing failed. Try changing the delimiter or header setting from the sidebar.")
        st.code(str(e))