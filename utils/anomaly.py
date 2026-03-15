# utils/anomaly.py
import pandas as pd
from sklearn.ensemble import IsolationForest

class AnomalyDetectionEngine:
    """
    Uses statistical methods to flag suspicious rows or values without deleting them.
    """
    def __init__(self):
         pass

    def detect_anomalies(self, df, policies):
         """
         Returns a dataframe of flagged anomalies and a list of anomaly metadata.
         """
         numeric_cols = []
         for col, policy in policies.get("column_policies", {}).items():
             if policy.get("flag_anomalies", False):
                 numeric_cols.append(col)
                 
         if not numeric_cols:
             return None, []
             
         numeric_df = df[numeric_cols].copy()
         numeric_df = numeric_df.dropna(axis=1, how="all")
         
         if numeric_df.shape[1] == 0 or numeric_df.shape[0] < 5:
             return None, []
             
         # Fill remaining NaNs just for the model (so it doesn't crash)
         numeric_df = numeric_df.fillna(numeric_df.median())
         if numeric_df.isnull().sum().sum() > 0: # If median was NaN
             numeric_df = numeric_df.fillna(0)
             
         # Ensure we don't train on millions of rows and stall Streamlit
         if numeric_df.shape[0] > 50000:
             sample_df = numeric_df.sample(n=50000, random_state=42)
         else:
             sample_df = numeric_df
             
         try:
             model = IsolationForest(contamination=0.05, random_state=42)
             model.fit(sample_df)
             preds = model.predict(numeric_df)
         except Exception:
             # In case of constant values or other sklearn failures
             return None, []
             
         result_df = df.copy()
         result_df["anomaly_flag"] = preds
         
         # -1 indicates anomaly
         anomalies = result_df[result_df["anomaly_flag"] == -1].copy()
         
         anomaly_metadata = []
         if not anomalies.empty:
             anomaly_metadata.append(f"Detected {len(anomalies)} statistical anomalies across numeric measures using Isolation Forest.")
             
         return anomalies, anomaly_metadata
