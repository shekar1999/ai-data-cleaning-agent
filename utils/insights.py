# utils/insights.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class InsightsEngine:
    """
    Visual insights and future integration point for LLM summaries.
    """
    def __init__(self):
        pass

    def generate_correlation_heatmap(self, df):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2:
            return None, "Not enough numeric columns for a correlation heatmap."
            
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax, fmt=".2f")
        return fig, None

    def generate_distribution_plot(self, df, column):
        valid_data = df[column].dropna()
        if valid_data.empty:
            return None, f"No valid data to plot for {column}."
            
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(valid_data, bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        return fig, None

    def get_llm_placeholder_summary(self, explainability_report):
        """
        Placeholder for future LLM integration. 
        Will eventually send the explainability_report to an LLM for natural language summary.
        """
        missing_reduced = explainability_report["summary"]["total_missing_before"] - explainability_report["summary"]["total_missing_after"]
        return f"[Future LLM Summary Hook] The engine analyzed {explainability_report['summary']['columns']} columns. It reduced missing values by {missing_reduced} and applied conservative imputation based on semantic inference. It preserved high-risk identifiers and lifecycle timestamps."
