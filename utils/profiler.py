# utils/profiler.py
import pandas as pd
import numpy as np

class DatasetProfiler:
    """
    Computes deep statistics for the dataset and individual columns.
    Used for inferring semantic meaning and cleaning policies.
    """
    def __init__(self):
        pass

    def profile(self, df):
        """
        Takes a DataFrame and returns a comprehensive metadata report.
        """
        row_count = df.shape[0]
        col_count = df.shape[1]
        
        report = {
            "row_count": row_count,
            "column_count": col_count,
            "total_missing": int(df.isnull().sum().sum()),
            "total_duplicate_rows": int(df.duplicated().sum()),
            "columns": {}
        }

        report["duplicate_ratio"] = report["total_duplicate_rows"] / max(1, row_count)
        
        # Take a max 50k row sample for expensive operations
        sample_df = df.sample(n=min(row_count, 50000), random_state=42) if row_count > 50000 else df
        
        for col in df.columns:
            series = df[col]
            missing_count = int(series.isnull().sum())
            missing_pct = missing_count / max(1, row_count)
            
            # Use sample for cardinality and distributions
            sample_series = sample_df[col]
            valid_sample = sample_series.dropna()
            
            unique_count = int(valid_sample.nunique())
            uniqueness_ratio = unique_count / max(1, len(valid_sample))
            
            dtype_str = str(series.dtype)
            
            col_profile = {
                "dtype": dtype_str,
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
                "unique_count": unique_count,
                "uniqueness_ratio": uniqueness_ratio,
            }

            if pd.api.types.is_numeric_dtype(series):
                col_profile["is_numeric"] = True
                col_profile["mean"] = float(valid_sample.mean()) if not valid_sample.empty else None
                col_profile["median"] = float(valid_sample.median()) if not valid_sample.empty else None
                col_profile["std"] = float(valid_sample.std()) if not valid_sample.empty else None
                col_profile["min"] = float(valid_sample.min()) if not valid_sample.empty else None
                col_profile["max"] = float(valid_sample.max()) if not valid_sample.empty else None
                
                # Outlier heuristic (3 sigma) on sample
                if not valid_sample.empty and col_profile["std"] and col_profile["std"] > 0:
                     outliers = valid_sample[np.abs(valid_sample - col_profile["mean"]) > 3 * col_profile["std"]]
                     # Scale up estimated outlier count to full dataset size
                     scale_factor = len(series.dropna()) / len(valid_sample) if len(valid_sample) > 0 else 1
                     col_profile["outlier_count"] = int(len(outliers) * scale_factor)
                else:
                     col_profile["outlier_count"] = 0
            else:
                col_profile["is_numeric"] = False
                
            # String parsing length on sample
            if dtype_str == 'object':
                str_lengths = valid_sample.astype(str).str.len()
                col_profile["mean_length"] = float(str_lengths.mean()) if not str_lengths.empty else None
                col_profile["max_length"] = float(str_lengths.max()) if not str_lengths.empty else None
            
            report["columns"][col] = col_profile
            
        return report
